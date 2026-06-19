# 失败案例库 (failure cases)

如实记录 citelocal-agent 在评估中**真实失败**的样本，作为「先有可复现的失败，再谈改进」的工作清单。不堆漂亮数字，只看哪里掉链子。

## 这批数据怎么来的

- 评测：`run_eval --split offline_sample`，**159 条**，跑在仓库自带的 **106 篇 `sample_notes/`** 语料上。
- 模型：答题 + 评判均为 `gpt-5.4-mini`(经 OpenAI 兼容网关)。
- **召回类失败是确定性的**(纯检索、无 LLM)，下面列出的 case 与头牌 README 表里那次同一来源,可精确复现。
- **答案 / 引用 / 拒答类失败依赖 LLM,输出有随机性**——下面的样本来自对 14 条拒答题的一次重跑,用来**例证失败模式**,具体哪几条命中会因 run 而异(聚合计数见 README:幻觉引用 4、拒答 13/14、out_of_scope 意图 0.80)。

复现：`python -m citelocal_agent.eval.run_eval --split offline_sample`(召回部分逐条打印,确定性)。

---

## 1. 检索召回失败(确定性,35 / 159 条 recall < 1.0)

召回在 `TOP_K=4` 下计算,口径是「一题所需的**每一段**证据都要命中」(严格 all-of)。这是全表最弱的一环,直接压住多跳答案质量的上限。

### 1a. 多跳题「漏掉第二篇」(recall 0.5,主导模式)

两篇必需文档只命中一篇。被漏掉的几乎总是**更宽泛的「枢纽型」笔记**,被问题里措辞更具体的那篇在排序上挤出 top-4。漏检最频繁的文档:

| 被漏掉的源文档 | 漏检次数 | 例子 case id |
|---|---|---|
| `transformers-and-attention.md` | 5 | `os_attention_vs_bm25`, `os_ann_dense_embeddings`, `os_mh_contextual_transformer`, `os_mh_pretrain_finetune_encoder` |
| `dense-and-sparse-retrieval.md` | 5 | `os_recall_ceiling_retrieval`, `os_mh_agentic_rag_retrieval`, `os_mh_eval_recall_hybrid` |
| `quantization.md` | 3 | `os_mh_qlora_quantization`, `os_mh_distill_vs_quant` |
| `evaluating-rag-systems.md` | 3 | `os_mh_tool_call_citations`, `os_mh_routing_agents` |
| `tokenization-and-embeddings.md` | 3 | `os_mh_embedding_dim_ann`, `os_mh_vectordb_stores_embeddings`, `os_mh_icl_token_cost` |

> 例:`os_mh_qlora_quantization`「QLoRA 如何把量化和低秩适配结合」——命中了 `qlora.md`,漏了 `quantization.md`(被 `qlora.md` 自己的相邻 chunk 占满名额)。

### 1b. 单源题彻底漏检(recall 0.0,5 条)

该题唯一该命中的文档没进 top-4,被主题相近的「兄弟笔记」整体盖过——这正是 README 吹「100+ 干扰文档仍稳召回」**翻车的反例**:

| case id | 类别 | 该命中却漏掉的文档 | 问题 |
|---|---|---|---|
| `os_rag_pipeline_steps` | single_paper | `retrieval-augmented-generation.md` | RAG 流水线主要阶段 |
| `os_per_category_reporting` | single_paper | `evaluating-rag-systems.md` | 为何按类别而非只看总体报指标 |
| `os_ft_fine_tuning` | definitional | `fine-tuning-and-transfer-learning.md` | 什么是微调、主要代价 |
| `os_quantization` | definitional | `quantization.md` | 什么是量化、权衡什么 |
| `os_ndcg` | definitional | `ndcg.md` | nDCG 比简单指标多算了什么 |

### 1c. 多跳题两篇全漏(recall 0.0,2 条)

`os_mh_lora_transformer`(漏 `fine-tuning-and-transfer-learning.md` + `transformers-and-attention.md`)、`os_mh_react_tool_calling`(漏 `react-reason-and-act.md` + `tool-calling-mechanics.md`)。

### 候选修法(待验证)

- 给多跳 / 一问多文的检索单独提高 `TOP_K` 或 `CANDIDATE_K`,而非全局 4。
- 重排阶段加**来源多样性**约束(同一文档的 chunk 不要占满名额),给第二篇源留位。
- 对 1b 这类「单文档被兄弟笔记盖过」,考虑查询改写 / 关键实体加权。

---

## 2. 答案 / 引用 / 拒答失败(LLM 侧,非确定性,代表性样本)

### 2a. 越界误判 + 编排把内部草稿标签当成引用(最严重)

- `os_oos_marathon`(out_of_scope,「怎么训练第一次马拉松」):被路由器**误判为 in_scope**,走了多智能体编排;答案文字其实拒答了,但引用列表里冒出 `verified findings: sub-question 1 / 2 / 3` 这种**编排内部草稿标签**,被引用校验判为无依据(幻觉引用)。
- 即:一次失败叠了两个 bug——**路由误判** + **引用卫生**(内部标签泄漏成对外引用)。

### 2b. 软拒答漏带事实

- `os_transformer_training_schedule`(no_answer,「原始 Transformer 的 warmup 调度具体是什么」):嘴上说「知识库没有具体调度」,却又陈述了「warmup 在前几千步线性升高学习率再衰减」并引 `learning-rate-warmup.md`。judge 判**拒答不合格**——既然该拒答,就不该顺带把相邻事实讲出来。

### 2c. 退化引用 token

- `os_chroma_install`(no_answer):答案正确拒答,但引用列表里塞了字面 `"none"`,被当成无依据引用计入幻觉。属于 `extract_outcome` 的引用解析没过滤掉非定位符 token。

### 2d. 编排递归爆顶

- `os_na_best_chunk_size`(no_answer):多智能体编排在 `RECURSION_LIMIT=12` 内未触发停止条件,直接 `GraphRecursionError` 整条失败。
- `os_gpt4_architecture`(no_answer):同样撞到递归上限,但靠「回退到简单路径」的兜底救回、最终正确拒答——说明**该拒答的简单题不该进编排**,进了还容易空转。

### 候选修法(待验证)

- **路由**:收紧 out_of_scope / no_answer 的判定,边界题宁可走简单路径(也便于干净拒答)。
- **引用卫生**:在 `extract_outcome` / 引用校验里过滤 `none`、`verified findings: …` 等非定位符,杜绝内部标签外泄。
- **拒答**:拒答时禁止顺带陈述相邻事实(2b)。
- **编排稳健性**:为「该拒答 / 简单」问题避免进编排;给编排更明确的停止条件,降低递归爆顶。

---

## 已知统计局限(别把小样本当定论)

- out_of_scope 仅 **5 条**、no_answer **11 条**(其中 `offline_sample` 9 条):100% / 0.80 这类数字统计意义有限。
- 答案正确率用 **LLM judge**,有评判偏差;judge prompt 见 `src/citelocal_agent/eval/prompts.py`,可抽样人工复核。
- 上述数字依赖 `LLM_MODEL`(本批为 `gpt-5.4-mini`),换模型应重跑。
