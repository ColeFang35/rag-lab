"""两个索引：FAISS 向量索引 + BM25 关键词索引。"""
from __future__ import annotations

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from .chunkers import Chunk


class VectorIndex:
    """余弦相似度检索。向量已 L2 归一化，所以用内积索引 IndexFlatIP。

    规模说明：这里用最朴素的暴力索引（IndexFlatIP）。几十到几万条它够快，
    而且结果精确、没有近似误差。上到百万级才需要换成 HNSW / IVF 这类
    近似最近邻索引，那时候要开始权衡召回率和延迟。
    """

    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if self.index.ntotal == 0:
            return []
        scores, idx = self.index.search(query_vec.reshape(1, -1), min(top_k, self.index.ntotal))
        return [(self.chunks[i], float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]


class BM25Index:
    """关键词检索。中文要先分词，否则整句被当成一个 token，BM25 就废了。"""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.tokens = [list(jieba.cut(c.text)) for c in chunks]
        self.bm25 = BM25Okapi(self.tokens) if self.tokens else None

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(list(jieba.cut(query)))
        order = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]
