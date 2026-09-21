#!/usr/bin/env python3
"""下载 ONNX 模型。默认走 hf-mirror.com（国内可直连，不需要代理）。

    python download_model.py            # 只下 embedding 模型（23MB，够跑 run_eval）
    python download_model.py --rerank   # 连 cross-encoder 重排模型一起下（266MB）
"""
from __future__ import annotations

import argparse
import pathlib
import urllib.request

MIRROR = "https://hf-mirror.com"
ROOT = pathlib.Path(__file__).parent / "models"

TARGETS = {
    "bge-small-zh-v1.5": [
        ("Xenova/bge-small-zh-v1.5", "onnx/model_quantized.onnx", "model_quantized.onnx"),
        ("Xenova/bge-small-zh-v1.5", "tokenizer.json", "tokenizer.json"),
        ("Xenova/bge-small-zh-v1.5", "tokenizer_config.json", "tokenizer_config.json"),
        ("Xenova/bge-small-zh-v1.5", "config.json", "config.json"),
    ],
    "bge-reranker-base": [
        ("Xenova/bge-reranker-base", "onnx/model_int8.onnx", "model_int8.onnx"),
        ("Xenova/bge-reranker-base", "tokenizer.json", "tokenizer.json"),
    ],
}


def fetch(repo: str, remote: str, dest: pathlib.Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  跳过（已存在） {dest.name}  {dest.stat().st_size/1048576:.0f}MB")
        return
    url = f"{MIRROR}/{repo}/resolve/main/{remote}"
    print(f"  下载 {dest.name} ...", end="", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    print(f" {dest.stat().st_size/1048576:.0f}MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true", help="连 cross-encoder 重排模型一起下")
    args = ap.parse_args()

    names = ["bge-small-zh-v1.5"] + (["bge-reranker-base"] if args.rerank else [])
    for name in names:
        print(f"{name}:")
        d = ROOT / name
        d.mkdir(parents=True, exist_ok=True)
        for repo, remote, out in TARGETS[name]:
            fetch(repo, remote, d / out)
    print("\n完成。")


if __name__ == "__main__":
    main()
