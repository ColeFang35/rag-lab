#!/usr/bin/env python3
"""第四层：推理框架对照 —— 同一个模型，ONNX Runtime 与 OpenVINO 差多少。

    python convert_to_openvino.py     # ONNX(int8) → IR
    python quantize_nncf.py           # FP32 → NNCF int8（可选，但比出来才公平）
    python bench_frameworks.py

## 为什么做这个

前三层回答的都是「**方案**选得对不对」（分块怎么切、要不要重排、检索用向量还是关键词）。
这一层换一个问法：**模型不动、数据不动，只换推理框架，会怎样？**

换框架在工程上常被默认成「免费的提速」。实测不是。这个脚本把三笔账分开算：

  1. **精度**：谁更接近 FP32 真模型？（用最小余弦衡量，不用绝对指标，避免偏袒任何一方）
  2. **延迟**：谁的稳态吞吐高？（要算上 reshape / 重编译这笔开销）
  3. **端到端**：换完之后检索指标变了吗？（这才是最终用户感知的）

## 四种组合

| 标签 | 模型来源 | 量化工具链 |
|---|---|---|
| `ONNX Runtime int8` | `model_quantized.onnx` | **onnxruntime** 自家 quantizer |
| `OV + onnxrt int8` | 同一个文件转成 IR | onnxruntime（OpenVINO 只是照跑） |
| `OV + NNCF int8` | 从 **FP32** 重新量化 | **NNCF**（OpenVINO 自家，用本仓库语料校准） |
| `OV + FP32` | FP32 原始模型 | 不量化（精度上限参照） |

**为什么必须有 NNCF 那一行**：只跟前两行比，等于让 OpenVINO 跑别人压出来的 int8，
是**不公平**的。NNCF 那一行才是「给 OpenVINO 一个公平机会」。
**为什么必须有 FP32 那一行**：没有真值参照，「谁更准」就无从谈起。

## 结论先说（本机 2026-09-29 实测，Intel Mac / CPU-only）

**只比前两行会得出错误结论。** 先做「ONNX Runtime vs OpenVINO（跑 onnxruntime 压的 int8）」时，
看着是「慢 2.07×、且 R@1 从 94% 掉到 89%」。补上 **NNCF 从 FP32 重量化**那一行之后：

  - 延迟 2.07× → **约 1.7×**（比跑外来的 int8 快约 20%；同一配置两次测量落在 1.66~1.84×）
  - 精度 0.9861 → **0.9871**，**比 onnxruntime 那版更接近 FP32**
  - 端到端 `R@1` **回到 94%**、MRR 复原、逐条排名 **0/18** 变化

**同一份权重、同一个 CPU，只换「谁来量化」，三件事一起变好。**
所以那个差距里有一大块来自**量化产物不匹配**，不是框架本身——
**做框架对比，量化工具链是必须先控制的变量。**

本机还有两个不利条件要说明：**老 Intel CPU 没有 VNNI/AMX**（OpenVINO 的优势恰好吃这些），
模型也小（23MB），框架调度开销占比被放大。**结论只适用于这个场景，不能外推。**
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import time

import numpy as np
import openvino as ov

from raglab.embedder import BGEEmbedder
from raglab.openvino_embedder import OpenVINOEmbedder
from raglab.chunkers import chunk_documents
from raglab.index import VectorIndex
from raglab.evaluate import evaluate
from raglab.retrieve import retrieve
from run_eval import load_docs

ROOT = pathlib.Path(__file__).parent
M = ROOT / "models"
ONNX_DIR = M / "bge-small-zh-v1.5"
IR_ONNXRUNTIME = M / "bge-small-zh-v1.5-ov"          # convert_to_openvino.py 产出
IR_NNCF = M / "bge-small-zh-v1.5-ov-nncf"            # quantize_nncf.py 产出
IR_FP32 = M / "bge-small-zh-v1.5-ov-fp32"            # quantize_nncf.py 产出

P = ov.properties
OV_CONFIGS = {
    "默认": {},
    "LATENCY": {P.hint.performance_mode: P.hint.PerformanceMode.LATENCY},
    "THROUGHPUT": {P.hint.performance_mode: P.hint.PerformanceMode.THROUGHPUT},
}


def bench_interleaved(fns: dict, rounds: int = 25, warmup: int = 5) -> dict:
    """**轮转交错**测多档耗时，返回每档的中位 / 最小 / 最大。

    ⚠️ 为什么要交错，而不是「一档测完再测下一档」：
        这台机器平时就跑着 VM（实测 ``com.apple.Virtualization.VirtualMachine``
        能吃到 500% CPU，load average 8+）。顺序测的话，某个瞬时尖峰只会砸在
        当时正在跑的那一档上，把它单独拉高——我就因此量出过 **40.5 ms/块**，
        而同一模型单独测只有 **9.7 ms/块**，差了 4 倍，全是假象。
        交错之后每轮的干扰被均摊到所有档，取中位才站得住。

    同时输出 min（最干净的一次）和 max（最差一次），让读者自己判断数据有多稳。
    """
    for _ in range(warmup):
        for fn in fns.values():
            fn()
    samples = {k: [] for k in fns}
    for _ in range(rounds):
        for k, fn in fns.items():
            t = time.perf_counter()
            fn()
            samples[k].append((time.perf_counter() - t) * 1000)
    return {k: {"median": statistics.median(v), "min": min(v), "max": max(v)}
            for k, v in samples.items()}


def cos_min(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).sum(axis=1).min())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=25, help="每项测多少次取中位")
    ap.add_argument("--no-e2e", action="store_true", help="跳过端到端检索对照")
    args = ap.parse_args()

    if not (IR_ONNXRUNTIME / "model.xml").exists():
        raise SystemExit("✗ 没找到 IR，先跑：python convert_to_openvino.py")

    docs = load_docs()
    queries = json.loads((ROOT / "data" / "eval" / "queries.json").read_text(encoding="utf-8"))["queries"]
    chunks = chunk_documents(docs, "heading")
    doc_texts = [c.text for c in chunks]
    q_texts = [q["query"] for q in queries]

    print(f"OpenVINO {ov.__version__}   设备 {ov.Core().available_devices}")
    print(f"知识库 {len(docs)} 篇 / {len(chunks)} 块   评测集 {len(queries)} 条 query\n")

    # 组合清单：(标签, 构造器, 是否 FP32 参照)
    variants: list[tuple[str, object, bool]] = [
        ("ONNX Runtime int8", lambda cfg=None: BGEEmbedder(str(ONNX_DIR)), False),
        ("OV + onnxrt int8", lambda cfg=None: OpenVINOEmbedder(str(IR_ONNXRUNTIME), config=cfg), False),
    ]
    if (IR_NNCF / "model.xml").exists():
        variants.append(("OV + NNCF int8",
                         lambda cfg=None: OpenVINOEmbedder(str(IR_NNCF), config=cfg), False))
    else:
        print("  ⚠️ 没找到 NNCF 量化模型，跳过该行（跑 python quantize_nncf.py 生成）")
    if (IR_FP32 / "model.xml").exists():
        variants.append(("OV + FP32", lambda cfg=None: OpenVINOEmbedder(str(IR_FP32), config=cfg), True))

    # ------------------------------------------------------------ 参照：FP32
    print("① 精度：各版本与 FP32 真模型的接近程度（越小越差）")
    fp32 = OpenVINOEmbedder(str(IR_FP32)) if (IR_FP32 / "model.xml").exists() else None
    ref_doc = ref_qry = None
    if fp32 is not None:
        ref_doc, ref_qry = fp32.encode_documents(doc_texts), fp32.encode_queries(q_texts)

    accuracy, built = {}, {}
    for label, make, is_ref in variants:
        emb = make()
        d, q = emb.encode_documents(doc_texts), emb.encode_queries(q_texts)
        built[label] = emb
        if is_ref:
            accuracy[label] = {"vs_fp32_cosine": 1.0, "note": "FP32 参照"}
            print(f"   {label:20} 参照（精度上限）")
        elif fp32 is not None:
            cd, cq = cos_min(ref_doc, d), cos_min(ref_qry, q)
            accuracy[label] = {"vs_fp32_doc": cd, "vs_fp32_query": cq, "min_cosine": min(cd, cq)}
            print(f"   {label:20} 最小余弦 {min(cd, cq):.6f}   （文档 {cd:.6f} / 查询 {cq:.6f}）")
        else:
            print(f"   {label:20} （没有 FP32 参照，跳过精度对比）")

    # ------------------------------------------------------------ ② 延迟
    # ⚠️ 所有档**交错**着测，不能一档测完再测下一档（本机常有 VM 抢 CPU，见 bench_interleaved 注释）
    print(f"\n② 延迟（全部 {len(doc_texts)} 个文档块编码一遍，batch=16，轮转交错取中位）")
    nncf_dir = IR_NNCF if (IR_NNCF / "model.xml").exists() else IR_ONNXRUNTIME

    runners = {}
    for label, make, is_ref in variants:
        if is_ref or label == "OV + FP32":
            continue
        runners[label] = built[label]
    # 追加 NNCF 的性能开关档
    for cfg_label, cfg in OV_CONFIGS.items():
        runners[f"NNCF int8 · {cfg_label}"] = OpenVINOEmbedder(str(nncf_dir), config=cfg)

    # 先把形状缓存填上，避免把编译时间算进稳态
    for e in runners.values():
        e.encode_documents(doc_texts)

    stats = bench_interleaved({k: (lambda e=e: e.encode_documents(doc_texts))
                               for k, e in runners.items()}, args.repeat)

    latency = {}
    base = stats["ONNX Runtime int8"]["median"]
    print(f"   {'配置':22}{'中位':>9}{'最快':>9}{'最差':>9}   ms/块    相对 ONNX")
    for k, s in stats.items():
        latency[k] = s
        rel = "—" if k == "ONNX Runtime int8" else f"{s['median']/base:.2f}×"
        print(f"   {k:22}{s['median']:8.1f}{s['min']:9.1f}{s['max']:9.1f}"
              f"   {s['median']/len(doc_texts):6.3f}   {rel}")

    # 冷启动：含 reshape + 重编译
    cold = OpenVINOEmbedder(str(IR_ONNXRUNTIME))
    t0 = time.perf_counter()
    cold.encode_documents(doc_texts)
    cold_ms = (time.perf_counter() - t0) * 1000
    print(f"\n   冷启动（含 reshape+重编译）  {cold_ms:8.2f} ms　{cold.recompiles} 次重编译")
    load = os.getloadavg()[0]
    print(f"   测量时系统负载 {load:.2f}"
          f"（本机核数有限，>核数说明有别的进程在抢，数字仅供参考）")

    # ------------------------------------------------------------ ③ 端到端
    e2e, flips = {}, {}
    if not args.no_e2e:
        print("\n③ 端到端检索（heading 分块 + 纯向量检索，只换 embedder）")
        indexes = {}
        for label, make, is_ref in variants:
            emb = built[label]
            vindex = VectorIndex(emb.dim)
            vindex.add(chunks, emb.encode_documents(doc_texts))
            indexes[label] = vindex
            res = evaluate(lambda q, k, e=emb, v=indexes[label]: [
                c for c, _ in retrieve("vector", q, vindex=v, bindex=None,
                                       embedder=e, reranker=None, top_k=k)],
                queries, (1, 3, 5))
            e2e[label] = {"R@1": round(res.recall_at(1), 4), "R@3": round(res.recall_at(3), 4),
                          "R@5": round(res.recall_at(5), 4), "MRR": round(res.mrr, 4),
                          "misses": res.misses}
            print(f"   {label:20} R@1 {res.recall_at(1):.0%}   R@3 {res.recall_at(3):.0%}"
                  f"   R@5 {res.recall_at(5):.0%}   MRR {res.mrr:.3f}")

        # ⚠️ 光看总体指标会漏掉最关键的信息：**它是均匀地差一点，还是个别条目翻车？**
        # 逐条比 gold 的排名，把翻掉的连同分差一起打出来。
        def gold_rank(label, q):
            hits = retrieve("vector", q["query"], vindex=indexes[label], bindex=None,
                            embedder=built[label], reranker=None, top_k=len(chunks))
            docs_ = [c.doc for c, _ in hits]
            return docs_.index(q["gold"]) + 1, hits

        base_label = "ONNX Runtime int8"
        for label, _, is_ref in variants:
            if label == base_label or is_ref:
                continue
            flips[label] = []
            for q in queries:
                r1, h1 = gold_rank(base_label, q)
                r2, h2 = gold_rank(label, q)
                if r1 != r2:
                    flips[label].append({
                        "query": q["query"], "gold": q["gold"],
                        "base_rank": r1, "rank": r2,
                        "base_top2": [[h1[0][0].doc, round(h1[0][1], 6)], [h1[1][0].doc, round(h1[1][1], 6)]],
                        "top2": [[h2[0][0].doc, round(h2[0][1], 6)], [h2[1][0].doc, round(h2[1][1], 6)]],
                    })
            print(f"\n   vs {base_label}：{label} —— 排名变化 {len(flips[label])} / {len(queries)} 条")
            for f in flips[label]:
                print(f"     ⚠ 「{f['query']}」 {f['base_rank']} → {f['rank']}")
                print(f"        基准前二 {' / '.join(f'{d}({s})' for d, s in f['base_top2'])}"
                      f"　分差 {f['base_top2'][0][1]-f['base_top2'][1][1]:+.6f}")
                print(f"        本档前二 {' / '.join(f'{d}({s})' for d, s in f['top2'])}"
                      f"　分差 {f['top2'][0][1]-f['top2'][1][1]:+.6f}")

    # ------------------------------------------------------------ 落盘
    out = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "openvino": ov.__version__,
        "device": "CPU",
        "docs": len(docs), "chunks": len(chunks), "queries": len(queries),
        "accuracy": accuracy,
        # 每档是 {median, min, max}——**保留 min**：本机常有 VM 抢 CPU，
        # min 是最不受干扰的一次，用来估计真实性能比中位更稳
        "latency_ms": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in latency.items()},
        "cold_start_ms": round(cold_ms, 2),
        "loadavg_at_measure": round(os.getloadavg()[0], 2),
        "end_to_end": e2e,
        "rank_flips": flips,
    }
    path = ROOT / "results" / f"framework-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果已写入 {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
