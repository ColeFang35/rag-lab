#!/usr/bin/env python3
"""分块策略 × 检索方式 的对照实验。

用法：
    python run_eval.py                 # 全部组合
    python run_eval.py --no-rerank     # 不加载 cross-encoder（省内存）
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import time

from raglab.chunkers import STRATEGIES, chunk_documents
from raglab.embedder import BGEEmbedder
from raglab.evaluate import evaluate
from raglab.index import BM25Index, VectorIndex
from raglab.retrieve import CrossEncoderReranker, retrieve

ROOT = pathlib.Path(__file__).parent
EMBED_DIR = ROOT / "models" / "bge-small-zh-v1.5"
RERANK_DIR = ROOT / "models" / "bge-reranker-base"


def load_docs() -> dict[str, str]:
    """把 markdown 按二级标题拆成一篇篇『文档』，标题即文档名。

    这样切的原因是评测需要细粒度的 ground truth：如果整份文件算一篇文档，
    一共只有 2 篇，recall 就失去意义了。
    """
    docs: dict[str, str] = {}
    for md in sorted((ROOT / "data" / "docs").glob("*.md")):
        text = md.read_text(encoding="utf-8")
        for block in re.split(r"\n(?=## )", text):
            m = re.match(r"##\s+(.+)", block.strip())
            if m:
                docs[m.group(1).strip()] = block.strip()
    return docs


def build(strategy: str, docs: dict[str, str], embedder: BGEEmbedder):
    chunks = chunk_documents(docs, strategy)
    t0 = time.time()
    vectors = embedder.encode_documents([c.text for c in chunks])
    embed_ms = (time.time() - t0) * 1000

    vindex = VectorIndex(embedder.dim)
    vindex.add(chunks, vectors)
    return chunks, vindex, BM25Index(chunks), embed_ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    docs = load_docs()
    queries = json.loads((ROOT / "data" / "eval" / "queries.json").read_text(encoding="utf-8"))["queries"]
    print(f"知识库：{len(docs)} 篇文档   评测集：{len(queries)} 条 query\n")

    embedder = BGEEmbedder(str(EMBED_DIR))
    reranker = None
    if not args.no_rerank and (RERANK_DIR / "model_int8.onnx").exists():
        t0 = time.time()
        reranker = CrossEncoderReranker(str(RERANK_DIR))
        print(f"cross-encoder 重排已加载（{time.time()-t0:.1f}s）\n")

    modes = ["vector", "bm25", "hybrid"] + (["hybrid+rerank"] if reranker else [])
    ks = (1, 3, 5)

    print(f"{'分块':<10}{'块数':>5}  {'检索方式':<15}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'MRR':>7}{'召回耗时':>10}")
    print("-" * 68)

    table = []
    for strategy in STRATEGIES:
        chunks, vindex, bindex, embed_ms = build(strategy, docs, embedder)

        for mode in modes:
            t0 = time.time()
            res = evaluate(
                lambda q, k, m=mode: [c for c, _ in retrieve(
                    m, q, vindex=vindex, bindex=bindex, embedder=embedder,
                    reranker=reranker, top_k=k)],
                queries, ks,
            )
            ms = (time.time() - t0) * 1000 / len(queries)
            print(f"{strategy:<10}{len(chunks):>5}  {mode:<15}"
                  f"{res.recall_at(1):>6.0%}{res.recall_at(3):>7.0%}{res.recall_at(5):>7.0%}"
                  f"{res.mrr:>7.3f}{ms:>9.0f}ms")
            table.append({"strategy": strategy, "chunks": len(chunks), "mode": mode,
                          "recall@1": round(res.recall_at(1), 4),
                          "recall@3": round(res.recall_at(3), 4),
                          "recall@5": round(res.recall_at(5), 4),
                          "mrr": round(res.mrr, 4),
                          "misses": res.misses})
        print()

    out = ROOT / "results" / f"eval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"docs": len(docs), "queries": len(queries), "table": table},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"结果已写入 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
