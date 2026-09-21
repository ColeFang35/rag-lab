#!/usr/bin/env python3
"""交互式检索：输入问题，看四种检索方式各自把哪条知识排在第一。

用法：
    python ask.py "出差住宿能报多少"
    python ask.py            # 进入交互模式
"""
from __future__ import annotations

import sys

import run_eval as R
from raglab.chunkers import chunk_documents
from raglab.embedder import BGEEmbedder
from raglab.index import BM25Index, VectorIndex
from raglab.retrieve import CrossEncoderReranker, retrieve


def main() -> None:
    docs = R.load_docs()
    embedder = BGEEmbedder(str(R.EMBED_DIR))

    # 演示统一用 heading 分块（评测里 R@1 最高）
    chunks = chunk_documents(docs, "heading")
    vindex = VectorIndex(embedder.dim)
    vindex.add(chunks, embedder.encode_documents([c.text for c in chunks]))
    bindex = BM25Index(chunks)

    reranker = None
    if (R.RERANK_DIR / "model_int8.onnx").exists() and (R.RERANK_DIR / "tokenizer.json").exists():
        reranker = CrossEncoderReranker(str(R.RERANK_DIR))

    modes = ["vector", "bm25", "hybrid"] + (["hybrid+rerank"] if reranker else [])
    queries = sys.argv[1:] or None

    def answer(q: str) -> None:
        print(f"\n问题：{q}")
        for mode in modes:
            hits = retrieve(mode, q, vindex=vindex, bindex=bindex,
                            embedder=embedder, reranker=reranker, top_k=3)
            top = hits[0][0].doc if hits else "（无）"
            print(f"  {mode:<15} top1 = {top}")
        if reranker:
            print("  → 重排的差别通常体现在第 2、3 条的顺序上，看 top1 变化不明显")

    if queries:
        for q in queries:
            answer(q)
    else:
        print("输入问题，回车看结果，Ctrl-C 退出。")
        while True:
            try:
                answer(input("\n> ").strip())
            except (EOFError, KeyboardInterrupt):
                break


if __name__ == "__main__":
    main()
