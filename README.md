# rag-lab · 最小可跑的 RAG 实验台

**分块 → 向量化 → 检索 → 重排 → 评测**，一条完整链路，全部在本机跑通。

做这个是因为原来的知识库检索是「关键词命中打分」（tags 命中权重 3、正文命中权重 1），
想搞清楚换成向量检索到底能好多少，以及**分块方式**对结果的影响有多大。

## 这条线里的位置

做 Agent 的过程中被同一类问题绊了几次，就各做了一个小实验：**这一步是真的对了，还是只是看起来对了。**

- **rag-lab**（本仓库）—— 检索方案哪个更好，重排值不值那个延迟
- [llm-judge-eval](https://github.com/ColeFang35/llm-judge-eval) —— 用 LLM 当裁判，裁判自己可不可信
- [decision-layer-lab](https://github.com/ColeFang35/decision-layer-lab) —— 模型自报的置信度能不能拿来设阈值

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

## 为什么用 ONNX 而不是 sentence-transformers

本机是 **Intel Mac，装不了 torch**。ONNX Runtime 在 x86 上一样跑，模型只用 23MB（int8 量化），
**不依赖 GPU、不依赖任何 API Key**，别人 clone 下来就能复现。

## 快速开始

```bash
pip install -r requirements.txt
python download_model.py     # 取 bge-small-zh-v1.5（约 23MB）
python run_eval.py           # 跑分块 × 检索的对照实验
python ask.py "出差住宿能报多少"
```

`run_eval.py` 会把每次结果写到 `results/eval-*.json`，包含没召回的 query。

## 目录

```
raglab/
  chunkers.py   三种分块：fixed（定长带重叠）/ heading（按标题）/ sentence（按句打包）
  embedder.py   ONNX 版 bge-small-zh-v1.5，mean pooling + L2 归一化
  index.py      FAISS 向量索引（IndexFlatIP）+ BM25（jieba 分词）
  retrieve.py   四种检索：vector / bm25 / hybrid(RRF) / hybrid+cross-encoder 重排
  evaluate.py   recall@k 与 MRR
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
