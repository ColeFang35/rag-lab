#!/usr/bin/env python3
"""用 NNCF 给 bge-small-zh 做 INT8 训练后量化（PTQ）。

    python quantize_nncf.py

## 为什么需要这个脚本

``bench_frameworks.py`` 里那个对比**对 OpenVINO 不公平**：它跑的是
**onnxruntime 自家 quantizer 压出来的 int8 产物**，OpenVINO 只是照跑别人的量化结果，
没用上自己的量化工具链和 kernel。

要给它一个公平的机会，就得**从 FP32 原始模型出发，用 NNCF（OpenVINO 的量化工具链）
按本仓库自己的语料重新量化一遍**，再和 onnxruntime 的 int8 比。

## 版本约束（Intel Mac 上是个死结，要绕）

- ``openvino`` 从 2026.1 起不再发布 macOS x86_64 的 wheel → 本机只能停在 **2025.4.1**
- 而 ``nncf`` 3.x 需要 OpenVINO 的新 2-bit 类型（``ov.Type.u2``），在 2025.4.1 上
  **import 就会报 `AttributeError: ... has no attribute 'u2'`**
- 解法是装**与 OpenVINO 2025.4.x 同一条发布线**的 **NNCF 2.19.0**（2025-12-01 发布）

**Linux / Windows 没有这个约束**，可以直接用最新的 OpenVINO + NNCF。
"""
from __future__ import annotations

import pathlib
import time
import urllib.request

import numpy as np
import nncf
import openvino as ov
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).parent
SRC_DIR = ROOT / "models" / "bge-small-zh-v1.5"       # 含 onnxruntime 压的 int8 + tokenizer
FP32_ONNX = SRC_DIR / "model.onnx"                     # FP32 原始模型（90MB，按需下载）
FP32_IR = ROOT / "models" / "bge-small-zh-v1.5-ov-fp32"
INT8_IR = ROOT / "models" / "bge-small-zh-v1.5-ov-nncf"

MIRROR = "https://hf-mirror.com"
FP32_URL = f"{MIRROR}/Xenova/bge-small-zh-v1.5/resolve/main/onnx/model.onnx"
MAX_LEN = 512
CALIB_BATCH = 8


def ensure_fp32() -> pathlib.Path:
    """FP32 原始模型（~90MB）。download_model.py 只取量化版，所以这里按需拉。"""
    if FP32_ONNX.exists() and FP32_ONNX.stat().st_size > 1_000_000:
        print(f"  FP32 模型已存在（{FP32_ONNX.stat().st_size/1048576:.0f}MB）")
        return FP32_ONNX
    print(f"  下载 FP32 模型（约 90MB）...", end="", flush=True)
    tmp = FP32_ONNX.with_suffix(".onnx.part")
    with urllib.request.urlopen(FP32_URL, timeout=900) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(FP32_ONNX)
    print(f" {FP32_ONNX.stat().st_size/1048576:.0f}MB")
    return FP32_ONNX


def build_calibration() -> list[dict[str, np.ndarray]]:
    """校准集：**用本仓库自己的语料**（知识库正文 + 评测 query）。

    ⚠️ 只有 30 多条，比 NNCF 文档建议的 100~300 条少很多——这是个务实的取舍：
    语料就这么大，与其去灌不相关的文本，不如用真实分布。小校准集的影响写进 README 了。
    """
    import json
    import sys
    sys.path.insert(0, str(ROOT))
    from raglab.chunkers import chunk_documents
    from run_eval import load_docs

    docs = load_docs()
    texts = [c.text for c in chunk_documents(docs, "heading")]
    texts += [q["query"] for q in json.loads(
        (ROOT / "data" / "eval" / "queries.json").read_text(encoding="utf-8"))["queries"]]

    tok = Tokenizer.from_file(str(SRC_DIR / "tokenizer.json"))
    tok.enable_truncation(max_length=MAX_LEN)
    encs = tok.encode_batch(texts)
    L = max(len(e.ids) for e in encs)

    ids = np.zeros((len(encs), L), dtype=np.int64)
    mask = np.zeros((len(encs), L), dtype=np.int64)
    for r, e in enumerate(encs):
        ids[r, : len(e.ids)] = e.ids
        mask[r, : len(e.attention_mask)] = e.attention_mask

    # 切成小 batch：校准是逐 batch 前向的，一次喂 34 条会吃很多内存
    samples = []
    for i in range(0, len(encs), CALIB_BATCH):
        samples.append({
            "input_ids": ids[i: i + CALIB_BATCH],
            "attention_mask": mask[i: i + CALIB_BATCH],
            "token_type_ids": np.zeros_like(ids[i: i + CALIB_BATCH]),
        })
    print(f"  校准集：{len(texts)} 条文本 → {len(samples)} 个 batch，序列长 {L}")
    return samples


def main() -> None:
    print(f"OpenVINO {ov.__version__}   NNCF {nncf.__version__}\n")
    core = ov.Core()

    # ---------------------------------------------------------- 1. FP32 → IR
    onnx = ensure_fp32()
    FP32_IR.mkdir(parents=True, exist_ok=True)
    fp32_xml = FP32_IR / "model.xml"
    t0 = time.time()
    model = core.read_model(str(onnx))
    ov.save_model(model, str(fp32_xml))
    print(f"  FP32 IR 写出（{time.time()-t0:.1f}s）")

    # ---------------------------------------------------------- 2. NNCF PTQ
    samples = build_calibration()
    dataset = nncf.Dataset(samples)
    t0 = time.time()
    quantized = nncf.quantize(
        model, dataset,
        preset=nncf.QuantizationPreset.PERFORMANCE,   # 面向吞吐；对称量化
        subset_size=len(samples),
        model_type=nncf.ModelType.TRANSFORMER,        # 告诉 NNCF 这是 transformer，跳过不适用的图变换
    )
    qtime = time.time() - t0
    INT8_IR.mkdir(parents=True, exist_ok=True)
    int8_xml = INT8_IR / "model.xml"
    ov.save_model(quantized, str(int8_xml))
    print(f"  NNCF 量化完成（{qtime:.1f}s）")

    # tokenizer 复制到**两个** IR 目录，让它们各自自包含
    # （⚠️ 漏了 FP32 那个会让 bench_frameworks.py 的 FP32 参照直接崩）
    for d in (FP32_IR, INT8_IR):
        for extra in ("tokenizer.json", "tokenizer_config.json", "config.json"):
            s = SRC_DIR / extra
            if s.exists():
                (d / extra).write_bytes(s.read_bytes())

    # ---------------------------------------------------------- 3. 体积对比
    def mb(p): return p.stat().st_size / 1048576
    print(f"\n  模型体积：")
    print(f"    onnxruntime int8 (ONNX)   {mb(SRC_DIR/'model_quantized.onnx'):6.1f} MB")
    print(f"    FP32 (ONNX)               {mb(onnx):6.1f} MB")
    print(f"    OpenVINO FP32 (IR)        {mb(fp32_xml)+mb(fp32_xml.with_suffix('.bin')):6.1f} MB")
    print(f"    OpenVINO NNCF int8 (IR)   {mb(int8_xml)+mb(int8_xml.with_suffix('.bin')):6.1f} MB")
    print(f"\n完成。跑 python bench_frameworks.py 看四种组合的对照。")


if __name__ == "__main__":
    main()
