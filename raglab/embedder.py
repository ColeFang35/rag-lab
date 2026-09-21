"""ONNX 版 bge-small-zh-v1.5 向量化。

为什么用 ONNX 而不是 sentence-transformers：本机是 Intel Mac，装不了 torch。
ONNX Runtime 在 x86 上一样能跑，模型只有 23MB（int8 量化），不依赖 GPU、不依赖任何 API Key。
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# bge 中文模型的检索指令前缀。查询侧要加，文档侧不加，这是官方训练时的用法。
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class BGEEmbedder:
    def __init__(self, model_dir: str, max_length: int = 512):
        self.tokenizer = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
        self.tokenizer.enable_truncation(max_length=max_length)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            f"{model_dir}/model_quantized.onnx", opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.output_name = self.session.get_outputs()[0].name
        self.dim = int(self.session.get_outputs()[0].shape[-1])

    def _forward(self, texts: list[str], batch_size: int) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encs = self.tokenizer.encode_batch(batch)
            maxlen = max(len(e.ids) for e in encs)
            ids = np.zeros((len(encs), maxlen), dtype=np.int64)
            mask = np.zeros((len(encs), maxlen), dtype=np.int64)
            for r, e in enumerate(encs):
                ids[r, : len(e.ids)] = e.ids
                mask[r, : len(e.attention_mask)] = e.attention_mask

            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run([self.output_name], feed)[0]

            # mean pooling：只对真实 token 求平均，padding 不参与
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            vectors.append(pooled.astype(np.float32))
        return np.vstack(vectors)

    def encode_documents(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        return self._normalize(self._forward(texts, batch_size))

    def encode_queries(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        prefixed = [QUERY_PREFIX + t for t in texts]
        return self._normalize(self._forward(prefixed, batch_size))

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        # L2 归一化之后，内积就等于余弦相似度，可以直接用 faiss.IndexFlatIP
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.clip(norms, 1e-9, None)
