# rag-lab · 最小可跑的 RAG 实验台

**分块 → 向量化 → 检索 → 重排 → 评测**，一条完整链路，全部在本机跑通。

做这个是因为原来的知识库检索是「关键词命中打分」（tags 命中权重 3、正文命中权重 1），
想搞清楚换成向量检索到底能好多少，以及**分块方式**对结果的影响有多大。

## 这条线里的位置

做 Agent 的过程中被同一类问题绊了几次，就各做了一个小实验：**这一步是真的对了，还是只是看起来对了。**

- **rag-lab**（本仓库）—— 检索方案哪个更好，重排值不值那个延迟
- [llm-judge-eval](https://github.com/ColeFang35/llm-judge-eval) —— 用 LLM 当裁判，裁判自己可不可信
- [decision-layer-lab](https://github.com/ColeFang35/decision-layer-lab) —— 模型自报的置信度能不能拿来设阈值
- **推理框架对照**（本仓库第四层）—— 同一个模型只换个推理框架，是不是就等于变快

## 结论先说

知识库 16 篇文档、评测集 18 条 query。**`heading` 分块 + 向量检索**：

| 分块 | 检索方式 | R@1 | R@3 | R@5 | MRR | 单条耗时 |
|---|---|---|---|---|---|---|
| heading | **vector** | **94%** | 100% | 100% | **0.972** | 3ms |
| sentence | vector | 94% | 100% | 100% | 0.972 | 3ms |
| fixed | vector | 89% | 100% | 100% | 0.944 | 3ms |
| fixed | hybrid+rerank | 89% | 100% | 100% | 0.944 | **679ms** |
| heading | hybrid+rerank | 83% | 100% | 100% | 0.917 | **600ms** |
| heading | hybrid(RRF) | 83% | 94% | 100% | 0.891 | 3ms |
| heading | bm25 | 78% | 83% | 83% | 0.806 | 0ms |

四个能拿去讲的点：

1. **向量检索比关键词检索 R@1 高 16 个点**（94% vs 78%），而且 **BM25 的 3 条漏检全是「用词和原文对不上」的口语问法**：
   「怎么开发票」「坐飞机是什么舱位标准」「临时有事来不及走流程怎么办」。
   这正是关键词检索的死穴，也是换向量的直接理由。
2. **分块方式确实影响结果**：`fixed`（固定长度切）比 `heading`（按标题切）低 5 个点，
   因为固定长度的切法会把一句话劈成两半，检索到的块缺上下文。
3. **混合检索在这个规模上没赢过纯向量**（R@1 83% vs 94%）。
   原因是 BM25 那一路拖了后腿，RRF 融合时把它的错误排名也带进来了。
   **这说明混合检索不是无脑更好，得看弱的那一路有多弱。**
4. **cross-encoder 重排要算成本账**：它把 hybrid 的 R@3 从 94% 拉到 100%、MRR 从 0.891 提到 0.917，
   但 **R@1 并没有提升**，而单条耗时从 **3ms 涨到 600ms，差 200 倍**。
   它还能补救差的分块（`fixed` 上 R@1 从 83% 回到 89%），但在这个规模上，
   **直接用好分块 + 纯向量（94% / 3ms）比重排更划算**。
   上了生产要考虑的是：先粗召回多少条、重排跑在什么硬件上、延迟预算允不允许。

> ⚠️ 数据规模说明：16 篇文档、18 条 query，是个验证链路用的**小样本**，
> 数字不能外推到生产。要说的是**方法**（怎么建评测、怎么对比），不是这几个百分比。

## 第四层：推理框架对照 —— 同一个模型，换个框架会怎样

前三层问的都是「**方案**选得对不对」（怎么切块、要不要重排、向量还是关键词）。
这一层换一个问法：**模型不动、数据不动，只换推理框架，会怎样。**

换框架在工程上常被默认成「免费的提速」。实测不是。同一个 int8 量化的 bge-small-zh-v1.5，
ONNX Runtime 与 **OpenVINO 2025.4.1**（CPU，本机 Intel Mac）对照：

| | ONNX Runtime | OpenVINO 默认 | OpenVINO `LATENCY` | OpenVINO `THROUGHPUT` |
|---|---|---|---|---|
| 向量化 16 个文档块（中位） | **108 ms** | 251 ms | 221 ms | 264 ms |
| 每块 | **6.8 ms** | 15.7 ms | 13.8 ms | 16.5 ms |
| 相对 ONNX | — | 2.32× | **2.04×** | 2.44× |

**数值**：两边输出最小余弦 **0.993**、`max|diff|` 1.9e-2（同一份 int8 权重，kernel 不同）。

**端到端**：同一套检索评测只换 embedder —— ONNX Runtime `R@1 94% / MRR 0.972`，
OpenVINO `R@1 89% / MRR 0.944`。**向量只差 0.7%，R@1 掉了 5 个点。**

### 四个能拿去讲的点

1. **动态形状是要付钱的。** 这个 ONNX 的输入是全动态 `[?, ?]`，而 OpenVINO 的 CPU 插件在动态维度上
   对 int8 量化节点做 shape 推断会直接报错（`PowerStatic` / `Eltwise shape infer mismatch`），
   必须 **reshape 成固定形状再编译**。后果是**每遇到一个新的 (batch, seq) 组合就要重新
   read + reshape + compile**（本机冷启动 882 ms，共 3 次重编译）。ONNX Runtime 的动态形状是原生的，
   不需要付这笔钱。**换框架的账要从这里开始算，不是从稳态吞吐开始算。**

2. **性能开关不是越多越好。** `THROUGHPUT` 模式在这个场景下**比默认还慢**（16.5 vs 15.7 ms/块）——
   它会把 stream 开起来，而 16 个块的小负载根本喂不饱，调度开销反而变成主要成本。
   `LATENCY` 才是这里该用的（13.8 ms/块，比默认快 12%）。**默认配置不等于上限，但也不等于下限。**

3. **⭐ 向量几乎一致，检索指标却掉了。** 这条最值得讲。两边向量最小余弦 0.993，
   看着"一样"，但 `R@1` 从 94% 掉到 89%。逐条查下来只有 1 条 query 翻了排名：

   ```
   「临时有事来不及走流程怎么办」
     ONNX  top1 紧急出差说明 (0.505635)   top2 七天无理由退货规则 (0.504505)   分差 +0.001130
     OV    top1 七天无理由退货规则 (0.508565)  top2 紧急出差说明 (0.505628)   分差 +0.002937
   ```

   **top1 和 top2 的分差只有 0.001 —— 这条 query 本来就是打平的。**
   在近乎打平的排序上，任何微小的数值差异都会翻名次。
   **所以「向量相似度 0.99 所以结果一样」这个推论是错的**，必须拿端到端指标说话。

4. **但这是个对 OpenVINO 不公平的对比，得说清楚。**
   - 那个 ONNX 模型**本来就是 onnxruntime 自家 quantizer 压出来的 int8**，
     OpenVINO 只是照跑别人的量化产物，**没用上自己的量化工具链（NNCF）和 kernel**；
   - 本机是**老 Intel CPU，没有 VNNI / AMX**，而 OpenVINO 的优势恰好吃这些指令集；
   - 模型太小（23MB），延迟在毫秒级，框架调度开销的占比被放大。

   **正确的说法不是「OpenVINO 不行」，而是「它在什么条件下才赢，我这个场景赢不了」。**

## 为什么用 ONNX 而不是 sentence-transformers

本机是 **Intel Mac，装不了 torch**。ONNX Runtime 在 x86 上一样跑，模型只用 23MB（int8 量化），
**不依赖 GPU、不依赖任何 API Key**，别人 clone 下来就能复现。

## 快速开始

```bash
pip install -r requirements.txt
python download_model.py     # 取 bge-small-zh-v1.5（约 23MB）
python run_eval.py           # 跑分块 × 检索的对照实验
python ask.py "出差住宿能报多少"

# 第四层（推理框架对照）
python convert_to_openvino.py   # ONNX → OpenVINO IR
python bench_frameworks.py      # ONNX Runtime vs OpenVINO：数值 / 延迟 / 端到端
```

`run_eval.py` 会把每次结果写到 `results/eval-*.json`，包含没召回的 query。
`bench_frameworks.py` 写到 `results/framework-*.json`，包含延迟与**排名发生变化的 query**。

⚠️ **Intel Mac 装 OpenVINO 要锁版本**：`openvino` 从 2026.1 起只发 macOS arm64 的 wheel，
x86_64 停在 **2025.4.1**。Linux / Windows 不受影响，可以装最新版。

## 目录

```
raglab/
  chunkers.py          三种分块：fixed（定长带重叠）/ heading（按标题）/ sentence（按句打包）
  embedder.py          ONNX 版 bge-small-zh-v1.5，mean pooling + L2 归一化
  openvino_embedder.py 同一模型的 OpenVINO 后端，接口与 embedder.py 一致，可直接替换
  index.py             FAISS 向量索引（IndexFlatIP）+ BM25（jieba 分词）
  retrieve.py          四种检索：vector / bm25 / hybrid(RRF) / hybrid+cross-encoder 重排
  evaluate.py          recall@k 与 MRR
convert_to_openvino.py ONNX → OpenVINO IR（保留动态形状，由 embedder 按需 reshape）
bench_frameworks.py    第四层：ONNX Runtime vs OpenVINO 的数值 / 延迟 / 端到端对照
data/
  docs/         知识库：云雀商城客服 FAQ（9 篇）+ 差旅报销政策（7 篇）
  eval/         评测集：18 条 query，标注答案所在小节
```

## 几个设计取舍

**为什么用 RRF 融合而不是加权求和**：余弦相似度在 0~1 之间，BM25 分数无上界，两者量纲没法直接比。
加权求和得先调权重、还要归一化；RRF 只看名次，`score = Σ 1/(k + rank)`，稳得多。

**为什么向量要先 L2 归一化**：归一化之后内积等于余弦相似度，就能直接用 `IndexFlatIP`，
不用自己写距离函数。

**为什么用 `IndexFlatIP` 暴力检索**：几十到几万条它够快，而且结果精确、没有近似误差。
上到百万级才需要换 HNSW / IVF，那时候才开始权衡召回率和延迟。**这里不做是因为不需要，不是因为不知道。**

**查询侧要加前缀**：bge 中文模型训练时查询侧带指令前缀
（`为这个句子生成表示以用于检索相关文章：`），文档侧不加。加了才对得上训练分布。

**cross-encoder 和 bi-encoder 的区别**：向量检索是 bi-encoder，query 和文档各自编码、只算一个余弦值，
快但粗；reranker 是 cross-encoder，把 query 和文档**拼在一起**送进模型，能建模两者的交互，准但慢
（每个候选都要跑一次前向）。所以标准做法是先用向量/BM25 粗召回一批，再用 cross-encoder 精排。

## 还没做的

- **文档解析**：知识库是直接写的 markdown，没处理过 PDF / Word 的版式、表格、扫描件 OCR
- **分块调优**：只比了三种切法，没系统扫过块大小和重叠长度的组合
- **向量库**：用的是 FAISS 本地索引，没用过 Qdrant / Milvus 这类服务化向量库
- **重排的延迟优化**：600ms 是 CPU 上 int8 模型逐条打分的结果，没做过 batch、量化再压、或者换更小的模型
- **生产规模**：16 篇文档，没在真实体量上验证过
- **NNCF 原生量化**：第四层用的是 onnxruntime 压出来的 int8 ONNX，OpenVINO 只是照跑。
  **没用 NNCF 在 OpenVINO 里重新量化过**，所以那个"慢 2 倍"的结果对 OpenVINO 不公平，不能当定论
- **bfloat16 / 动态量化**：没试过，也没试过更大模型——OpenVINO 的优势场景（大模型、带 VNNI/AMX 的
  新 Intel CPU、视觉与 LLM）一个都没覆盖到，本机是 CPU-only 的老 Intel Mac
