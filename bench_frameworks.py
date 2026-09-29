#!/usr/bin/env python3
"""第四层：推理框架对照 —— 同一个模型，ONNX Runtime 与 OpenVINO 差多少。

    python convert_to_openvino.py     # 先转 IR
    python bench_frameworks.py

## 为什么做这个

前三层回答的都是「**方案**选得对不对」（分块怎么切、要不要重排、检索用向量还是关键词）。
这一层换一个问法：**同一个模型、同一份数据，只换推理框架，会怎样？**

换框架在工程上常被当成「免费的提速」，实测下来不是。这个脚本把三笔账分开算：

  1. **数值**：两边输出一致吗？（不一致就是换错了）
  2. **延迟**：谁的稳态吞吐高？（要算上 reshape / 重编译这笔开销）
  3. **端到端**：换完之后检索指标变了吗？（这才是最终用户感知的）

## 结论先说（本机 2026-09-29 实测，Intel Mac / CPU-only）

**这个场景下 ONNX Runtime 更快，而且快得不是一点。** 但这是个**对 OpenVINO 不公平**的对比：

  - 那个 ONNX 模型**本来就是 onnxruntime 自家 quantizer 压出来的 int8**，
    OpenVINO 只是照跑别人的量化产物，没用上自己的量化工具链（NNCF）和 kernel；
  - 本机是**老 Intel CPU，没有 VNNI/AMX**，而 OpenVINO 的优势恰恰吃这些指令集；
  - 模型太小（23MB），延迟在毫秒级，框架的调度开销占比被放大。

**所以正确的说法不是「OpenVINO 不行」，而是「它在什么条件下才赢，我这个场景赢不了」。**
"""
from __future__ import annotations

import argparse
import json
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
ONNX_DIR = ROOT / "models" / "bge-small-zh-v1.5"
OV_DIR = ROOT / "models" / "bge-small-zh-v1.5-ov"

P = ov.properties
OV_CONFIGS = {
    "默认": {},
    "LATENCY": {P.hint.performance_mode: P.hint.PerformanceMode.LATENCY},
    "THROUGHPUT": {P.hint.performance_mode: P.hint.PerformanceMode.THROUGHPUT},
}


def bench(fn, n: int = 30, warmup: int = 5) -> float:
    """返回中位耗时（毫秒）。用中位数而不是均值，避免偶发调度抖动带偏结论。"""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=30, help="每项测多少次取中位")
    ap.add_argument("--no-e2e", action="store_true", help="跳过端到端检索对照")
    args = ap.parse_args()

    if not (OV_DIR / "model.xml").exists():
        raise SystemExit("✗ 没找到 IR 模型，先跑：python convert_to_openvino.py")

    docs = load_docs()
    queries = json.loads((ROOT / "data" / "eval" / "queries.json").read_text(encoding="utf-8"))["queries"]
    chunks = chunk_documents(docs, "heading")
    doc_texts = [c.text for c in chunks]
    q_texts = [q["query"] for q in queries]

    print(f"OpenVINO {ov.__version__}   设备 {ov.Core().available_devices}")
    print(f"知识库 {len(docs)} 篇 / {len(chunks)} 块   评测集 {len(queries)} 条 query\n")

    ort_emb = BGEEmbedder(str(ONNX_DIR))

    # ------------------------------------------------------------ ① 数值一致性
    print("① 数值一致性（同一批文本，两边输出对比）")
    base_doc = ort_emb.encode_documents(doc_texts)
    base_qry = ort_emb.encode_queries(q_texts)
    numerically = {}
    for label, cfg in OV_CONFIGS.items():
        ov_emb = OpenVINOEmbedder(str(OV_DIR), config=cfg)
        d = ov_emb.encode_documents(doc_texts)
        q = ov_emb.encode_queries(q_texts)
        cos_d = float((base_doc * d).sum(axis=1).min())
        cos_q = float((base_qry * q).sum(axis=1).min())
        mx = float(max(np.abs(base_doc - d).max(), np.abs(base_qry - q).max()))
        numerically[label] = {"min_cosine": min(cos_d, cos_q), "max_abs_diff": mx,
                              "recompiles": ov_emb.recompiles}
        print(f"   {label:11} 最小余弦 {min(cos_d, cos_q):.8f}   max|diff| {mx:.3e}"
              f"   重编译 {ov_emb.recompiles} 次")

    # ------------------------------------------------------------ ② 延迟
    print("\n② 延迟（把全部 {n} 个文档块编码一遍，batch=16，取中位）".format(n=len(doc_texts)))
    latency = {}
    ms = bench(lambda: ort_emb.encode_documents(doc_texts), args.repeat)
    latency["ONNX Runtime"] = ms
    print(f"   {'ONNX Runtime':14} {ms:8.2f} ms   {ms/len(doc_texts):6.3f} ms/块")

    for label, cfg in OV_CONFIGS.items():
        ov_emb = OpenVINOEmbedder(str(OV_DIR), config=cfg)
        ov_emb.encode_documents(doc_texts)          # 先跑一遍把形状编译缓存填上
        ms = bench(lambda: ov_emb.encode_documents(doc_texts), args.repeat)
        latency[f"OpenVINO {label}"] = ms
        print(f"   {'OpenVINO '+label:14} {ms:8.2f} ms   {ms/len(doc_texts):6.3f} ms/块"
              f"   （相对 ONNX {ms/latency['ONNX Runtime']:.2f}×）")

    # 冷启动：含 reshape + 重编译
    cold = OpenVINOEmbedder(str(OV_DIR))
    t0 = time.perf_counter()
    cold.encode_documents(doc_texts)
    cold_ms = (time.perf_counter() - t0) * 1000
    print(f"   {'OpenVINO 冷启动':14} {cold_ms:8.2f} ms   （含 {cold.recompiles} 次 reshape+重编译）")

    # ------------------------------------------------------------ ③ 端到端
    e2e, flips = {}, []
    if not args.no_e2e:
        print("\n③ 端到端检索（heading 分块 + 纯向量检索，只换 embedder）")
        ov_emb = OpenVINOEmbedder(str(OV_DIR))
        built = {}
        for label, emb in (("ONNX Runtime", ort_emb), ("OpenVINO 默认", ov_emb)):
            vectors = emb.encode_documents(doc_texts)
            vindex = VectorIndex(emb.dim)
            vindex.add(chunks, vectors)
            built[label] = (emb, vindex)
            res = evaluate(lambda q, k, e=emb, v=vindex: [
                c for c, _ in retrieve("vector", q, vindex=v, bindex=None,
                                       embedder=e, reranker=None, top_k=k)],
                queries, (1, 3, 5))
            e2e[label] = {"R@1": round(res.recall_at(1), 4), "R@3": round(res.recall_at(3), 4),
                          "R@5": round(res.recall_at(5), 4), "MRR": round(res.mrr, 4),
                          "misses": res.misses}
            print(f"   {label:14} R@1 {res.recall_at(1):.0%}   R@3 {res.recall_at(3):.0%}"
                  f"   R@5 {res.recall_at(5):.0%}   MRR {res.mrr:.3f}")

        # ⚠️ 光看总体指标会漏掉最关键的信息：**它是均匀地差一点，还是个别条目翻车？**
        # 逐条比 gold 的排名，把翻掉的那几条连同分差一起打出来。
        def gold_rank(label, q):
            emb, vindex = built[label]
            hits = retrieve("vector", q["query"], vindex=vindex, bindex=None,
                            embedder=emb, reranker=None, top_k=len(chunks))
            docs_ = [c.doc for c, _ in hits]
            return docs_.index(q["gold"]) + 1, hits

        for q in queries:
            r1, h1 = gold_rank("ONNX Runtime", q)
            r2, h2 = gold_rank("OpenVINO 默认", q)
            if r1 != r2:
                flips.append({
                    "query": q["query"], "gold": q["gold"],
                    "onnx_rank": r1, "ov_rank": r2,
                    "onnx_top2": [[h1[0][0].doc, round(h1[0][1], 6)],
                                  [h1[1][0].doc, round(h1[1][1], 6)]],
                    "ov_top2": [[h2[0][0].doc, round(h2[0][1], 6)],
                                [h2[1][0].doc, round(h2[1][1], 6)]],
                })

        same = e2e["ONNX Runtime"]["R@1"] == e2e["OpenVINO 默认"]["R@1"]
        print(f"   R@1 是否一致：{'是' if same else '否'}"
              f"　|　排名发生变化的 query：{len(flips)} / {len(queries)}")
        for f in flips:
            print(f"     ⚠ 「{f['query']}」 {f['onnx_rank']} → {f['ov_rank']}")
            print(f"        ONNX 前二 {' / '.join(f'{d}({s})' for d, s in f['onnx_top2'])}"
                  f"　分差 {f['onnx_top2'][0][1]-f['onnx_top2'][1][1]:+.6f}")
            print(f"        OV   前二 {' / '.join(f'{d}({s})' for d, s in f['ov_top2'])}"
                  f"　分差 {f['ov_top2'][0][1]-f['ov_top2'][1][1]:+.6f}")

    # ------------------------------------------------------------ 落盘
    out = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "openvino": ov.__version__,
        "device": "CPU",
        "docs": len(docs), "chunks": len(chunks), "queries": len(queries),
        "numerics": numerically,
        "latency_ms": {k: round(v, 3) for k, v in latency.items()},
        "cold_start_ms": round(cold_ms, 2),
        "end_to_end": e2e,
        "rank_flips": flips,
    }
    path = ROOT / "results" / f"framework-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果已写入 {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
