"""检索：向量 / BM25 / 混合（RRF） / 混合 + cross-encoder 重排。

四种方式的取舍，就是评测脚本要回答的问题。
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .chunkers import Chunk
from .index import BM25Index, VectorIndex


def vector_search(vindex: VectorIndex, embedder, query: str, top_k: int = 5):
    return vindex.search(embedder.encode_queries([query])[0], top_k)


def bm25_search(bindex: BM25Index, query: str, top_k: int = 5):
    return bindex.search(query, top_k)


def hybrid_rrf(vindex, bindex, embedder, query: str, top_k: int = 5, k: int = 60):
    """RRF（Reciprocal Rank Fusion）：按名次融合，不试图把两种分数归一化。

    为什么用 RRF 而不是加权求和：余弦相似度在 0~1 之间，BM25 是无上界的，
    两者量纲没法直接比。加权求和要先调权重、还要归一化，RRF 只看名次，稳得多。
    公式：score(d) = Σ 1 / (k + rank_i(d))，k 一般取 60。
    """
    pool = 20
    vec = vector_search(vindex, embedder, query, pool)
    bm = bm25_search(bindex, query, pool)

    fused: dict[str, float] = {}
    chosen: dict[str, Chunk] = {}
    for results in (vec, bm):
        for rank, (chunk, _) in enumerate(results, start=1):
            fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (k + rank)
            chosen[chunk.id] = chunk

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [(chosen[cid], score) for cid, score in ranked]


class CrossEncoderReranker:
    """bge-reranker-base（ONNX，int8）。

    和上面几个检索器的本质区别：它是 cross-encoder，把 query 和候选**拼在一起**
    送进模型，能建模两者的交互；而向量检索是 bi-encoder，query 和文档各自编码、
    只算一个余弦值，快但粗。所以标准做法是先用向量/BM25 粗召回一批，
    再用 cross-encoder 精排。代价是慢：每个候选都要跑一次前向。
    """

    def __init__(self, model_dir: str, max_length: int = 512):
        self.tokenizer = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
        self.tokenizer.enable_truncation(max_length=max_length)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            f"{model_dir}/model_int8.onnx", opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.output_name = self.session.get_outputs()[0].name

    def rerank(self, query: str, candidates: list[tuple[Chunk, float]], top_k: int = 5):
        if not candidates:
            return []
        pairs = [(query, c.text) for c, _ in candidates]
        encs = self.tokenizer.encode_batch(pairs)
        maxlen = max(len(e.ids) for e in encs)
        ids = np.zeros((len(encs), maxlen), dtype=np.int64)
        mask = np.zeros((len(encs), maxlen), dtype=np.int64)
        for r, e in enumerate(encs):
            ids[r, : len(e.ids)] = e.ids
            mask[r, : len(e.attention_mask)] = e.attention_mask

        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        logits = self.session.run([self.output_name], feed)[0]

        scores = logits.reshape(len(pairs), -1)[:, 0]  # reranker 输出单值 relevance
        order = np.argsort(scores)[::-1][:top_k]
        return [(candidates[i][0], float(scores[i])) for i in order]


RETRIEVERS = ("vector", "bm25", "hybrid", "hybrid+rerank")


def retrieve(mode: str, query: str, *, vindex, bindex, embedder, reranker=None,
             top_k: int = 5):
    if mode == "vector":
        return vector_search(vindex, embedder, query, top_k)
    if mode == "bm25":
        return bm25_search(bindex, query, top_k)
    if mode == "hybrid":
        return hybrid_rrf(vindex, bindex, embedder, query, top_k)
    if mode == "hybrid+rerank":
        coarse = hybrid_rrf(vindex, bindex, embedder, query, top_k=max(top_k * 4, 20))
        if reranker is None:
            return coarse[:top_k]
        return reranker.rerank(query, coarse, top_k)
    raise ValueError(f"unknown retrieval mode: {mode}")
