"""分块策略。

三种切法，用来回答同一个问题：块怎么切，检索效果差多少？
  - fixed    ：按固定字符数切、带重叠。简单，但会把一句话劈两半。
  - heading  ：按 markdown 标题切，一个标题一块。块大小天然不均，但语义完整。
  - sentence ：按句子打包到接近目标长度。介于两者之间。

评测脚本会把三种切法各跑一遍，用同一套 query 比 recall。
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    doc: str          # 来源文档名（评测的 ground truth 打在这一层）
    idx: int          # 块在文档内的序号
    text: str

    @property
    def id(self) -> str:
        return f"{self.doc}#{self.idx}"


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；!?;])\s*", text.strip())
    return [p.strip() for p in parts if p.strip()]


def fixed(text: str, doc: str, size: int = 180, overlap: int = 40) -> list[Chunk]:
    text = text.strip()
    chunks, i, n = [], 0, 0
    while i < len(text):
        piece = text[i : i + size].strip()
        if piece:
            chunks.append(Chunk(doc, n, piece))
            n += 1
        i += max(1, size - overlap)
    return chunks


def heading(text: str, doc: str) -> list[Chunk]:
    """按 markdown 二级标题切。标题行本身拼回块里，否则检索会丢掉标题信息。"""
    blocks = re.split(r"\n(?=#{1,3}\s)", text.strip())
    chunks = []
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        # 顶级标题（# 文档名）不单独成块
        if b.startswith("# ") and "\n##" not in b and len(b) < 60:
            continue
        chunks.append(Chunk(doc, len(chunks), b))
    return chunks


def sentence(text: str, doc: str, target: int = 180) -> list[Chunk]:
    chunks, buf, cur = [], [], 0
    for s in _split_sentences(text):
        if cur + len(s) > target and buf:
            chunks.append(Chunk(doc, len(chunks), "".join(buf).strip()))
            buf, cur = [], 0
        buf.append(s)
        cur += len(s)
    if buf:
        chunks.append(Chunk(doc, len(chunks), "".join(buf).strip()))
    return [c for c in chunks if c.text]


STRATEGIES = {
    "fixed": fixed,
    "heading": heading,
    "sentence": sentence,
}


def chunk_documents(docs: dict[str, str], strategy: str, **kw) -> list[Chunk]:
    fn = STRATEGIES[strategy]
    out: list[Chunk] = []
    for name, text in docs.items():
        out.extend(fn(text, name, **kw))
    return out
