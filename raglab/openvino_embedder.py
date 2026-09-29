"""OpenVINO 版 bge-small-zh-v1.5 向量化。

接口与 :class:`raglab.embedder.BGEEmbedder` **完全一致**
（``encode_documents`` / ``encode_queries`` / ``dim``），所以可以直接替换进 ``run_eval.py``，
用来回答一个问题：**同一个模型、同一份数据，换一个推理框架会怎样。**

## 为什么不能 read_model 之后直接跑

这个 ONNX 模型的输入是**全动态**的（``[?, ?]``）。OpenVINO 的 CPU 插件在动态维度上
对 int8 量化节点做 shape 推断会失败：

    [CPU] PowerStatic node 'Multiply_9857' Check 'input_shape[j] == 1' failed
    Eltwise shape infer input shapes dim index: 0 mismatch

所以必须 **reshape 成固定形状再编译**。代价是：**每遇到一个新的 (batch, seq_len)
组合就要重新 read + reshape + compile 一次**。ONNX Runtime 那边不需要付这笔钱
（它的动态形状是原生的）。

本模块按形状缓存已编译的模型，所以稳态下只付一次；但**第一批数据的首字延迟会明显更高**。
这是「换框架」在真实工程里首先要算的账，不是可以忽略的噪声。

## 兼容性

``openvino`` 从 **2026.1 起不再发布 macOS x86_64 的 wheel**（只剩 arm64）。
Intel Mac 上要用 **2025.4.1**（该版本提供 cp310–cp314 的 ``macosx_10_15_x86_64``）。
Linux / Windows 不受影响，可以装最新版。
"""
from __future__ import annotations

import pathlib

import numpy as np
import openvino as ov
from tokenizers import Tokenizer

# 与 embedder.py 保持一致：查询侧要加指令前缀，文档侧不加。
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class OpenVINOEmbedder:
    """OpenVINO 后端，接口对齐 BGEEmbedder。"""

    def __init__(self, model_dir: str, max_length: int = 512,
                 device: str = "CPU", config: dict | None = None,
                 pad_to: int | None = None):
        """
        Args:
            model_dir: 含 ``model.xml`` / ``model.bin`` / ``tokenizer.json`` 的目录
                       （由 ``convert_to_openvino.py`` 生成）。
            device: OpenVINO 设备名，本机 CPU-only 所以是 ``"CPU"``。
            config: OpenVINO 编译属性，例如 ``{ov.properties.hint.performance_mode:
                    ov.properties.hint.PerformanceMode.LATENCY}``。
            pad_to: 把所有输入补齐到固定长度。**固定形状可以复用同一份编译结果**，
                    用来把「推理」和「reshape/重编译」两种开销分开测。
        """
        self.dir = pathlib.Path(model_dir)
        self.max_length = max_length
        self.pad_to = pad_to
        self.config = dict(config or {})
        self.core = ov.Core()
        self.tokenizer = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)

        # reshape 是破坏性的，所以每次换形状都从磁盘重新读一份干净的模型
        base = self.core.read_model(str(self.dir / "model.xml"))
        out_shape = base.output(0).partial_shape
        self.dim = int(out_shape[-1].get_length())
        self._input_names = [i.any_name for i in base.inputs]

        self._cache: dict[tuple[int, int], object] = {}
        self.recompiles = 0          # 供基准测试统计：重编译次数

    # ------------------------------------------------------------------ 形状管理
    def _compiled(self, batch: int, length: int):
        key = (batch, length)
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        model = self.core.read_model(str(self.dir / "model.xml"))
        model.reshape({model.input(n): [batch, length] for n in self._input_names})
        compiled = self.core.compile_model(model, "CPU", self.config)
        self._cache[key] = compiled
        self.recompiles += 1
        return compiled

    # ------------------------------------------------------------------ 前向
    def _forward(self, texts: list[str], batch_size: int) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            encs = self.tokenizer.encode_batch(batch)
            maxlen = max(len(e.ids) for e in encs)
            if self.pad_to:
                maxlen = max(maxlen, self.pad_to)

            ids = np.zeros((len(encs), maxlen), dtype=np.int64)
            mask = np.zeros((len(encs), maxlen), dtype=np.int64)
            for r, e in enumerate(encs):
                ids[r, : len(e.ids)] = e.ids
                mask[r, : len(e.attention_mask)] = e.attention_mask

            compiled = self._compiled(len(encs), maxlen)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                # ⚠️ 这个模型有 token_type_ids 输入，必须显式喂，不能省
                feed["token_type_ids"] = np.zeros_like(ids)

            hidden = compiled(feed)[compiled.output(0)]

            # mean pooling：只对真实 token 求平均，padding 不参与（与 embedder.py 一致）
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
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.clip(norms, 1e-9, None)
