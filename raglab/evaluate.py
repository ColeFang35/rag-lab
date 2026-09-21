"""检索评测：recall@k 与 MRR。

为什么是这个两个指标：
  - recall@k 回答"该找的东西有没有进前 k 条"，直接决定下游大模型有没有材料可用；
  - MRR 看"正确的那条排在第几"，衡量排序质量，只有一条正确答案的场景特别合适。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalResult:
    n: int = 0
    recall: dict[int, int] = field(default_factory=dict)
    rr_sum: float = 0.0
    misses: list[str] = field(default_factory=list)

    def recall_at(self, k: int) -> float:
        return self.recall.get(k, 0) / self.n if self.n else 0.0

    @property
    def mrr(self) -> float:
        return self.rr_sum / self.n if self.n else 0.0


def evaluate(retrieve_fn, queries: list[dict], ks=(1, 3, 5)) -> EvalResult:
    """retrieve_fn(query, top_k) -> list[Chunk]（按相关度降序）"""
    res = EvalResult()
    max_k = max(ks)

    for item in queries:
        q, gold = item["query"], item["gold"]
        hits = retrieve_fn(q, max_k)
        docs = [c.doc for c in hits]

        res.n += 1
        for k in ks:
            if gold in docs[:k]:
                res.recall[k] = res.recall.get(k, 0) + 1

        rank = next((i for i, d in enumerate(docs, 1) if d == gold), None)
        if rank:
            res.rr_sum += 1.0 / rank
        else:
            res.misses.append(q)

    return res
