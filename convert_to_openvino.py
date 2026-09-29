#!/usr/bin/env python3
"""把 ONNX 模型转成 OpenVINO IR。

    python convert_to_openvino.py                 # 转 embedding 模型
    python convert_to_openvino.py --rerank        # 连 cross-encoder 一起转

产物落在 ``models/<模型名>-ov/``（``models/`` 在 .gitignore 里，IR 是派生物，不进仓库——
别人 clone 下来跑这个脚本就能重建）。

⚠️ 这里**故意不 reshape**：转出来保留动态形状，由 ``OpenVINOEmbedder`` 按实际输入形状
再 reshape 后编译。原因见 ``raglab/openvino_embedder.py`` 的模块文档——
这个模型的 int8 量化节点在动态维度上推不出 shape，必须固定形状才能跑。
"""
from __future__ import annotations

import argparse
import pathlib
import time

import openvino as ov

ROOT = pathlib.Path(__file__).parent
MODELS = ROOT / "models"

TARGETS = {
    "bge-small-zh-v1.5": "model_quantized.onnx",
    "bge-reranker-base": "model_int8.onnx",
}


def convert(name: str, onnx_file: str) -> None:
    src = MODELS / name / onnx_file
    if not src.exists():
        print(f"  跳过 {name}：找不到 {src.relative_to(ROOT)}（先跑 download_model.py）")
        return

    dst_dir = MODELS / f"{name}-ov"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "model.xml"

    core = ov.Core()
    t0 = time.time()
    model = core.read_model(str(src))
    ov.save_model(model, str(dst))
    dt = time.time() - t0

    bin_f = dst.with_suffix(".bin")
    size = lambda p: p.stat().st_size / 1048576
    print(f"  {name}:")
    print(f"    ONNX  {src.stat().st_size/1048576:6.1f} MB")
    print(f"    IR    {size(dst):6.1f} MB (.xml) + {size(bin_f):6.1f} MB (.bin)"
          f" = {size(dst)+size(bin_f):.1f} MB")
    print(f"    转换耗时 {dt*1000:.0f} ms")

    # 转完立刻编译一次确认能用（不 reshape 会在这里报 shape 错，属预期）
    for i in model.inputs:
        print(f"    输入 {i.any_name:18} {i.partial_shape}")
    print(f"    → 形状动态，需由 OpenVINOEmbedder reshape 后编译（见模块文档）\n")

    # tokenizer 也复制一份，让 -ov 目录自包含
    for extra in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        s = MODELS / name / extra
        if s.exists():
            (dst_dir / extra).write_bytes(s.read_bytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true", help="连 cross-encoder 重排模型一起转")
    args = ap.parse_args()

    names = ["bge-small-zh-v1.5"] + (["bge-reranker-base"] if args.rerank else [])
    print(f"OpenVINO {ov.__version__}   可用设备 {ov.Core().available_devices}\n")
    for n in names:
        convert(n, TARGETS[n])
    print("完成。跑 python bench_frameworks.py 看对照结果。")


if __name__ == "__main__":
    main()
