# 失败案例库 (failure cases)

如实记录 citelocal-agent 在评估中**真实失败**的样本，作为「先有可复现的失败，再谈改进」的工作清单。不堆漂亮数字，只看哪里掉链子。

## 这批数据怎么来的

- 评测：`run_eval --split offline_sample`，**159 条**，跑在仓库自带的 **106 篇 `sample_notes/`** 语料上。
- 模型：答题 + 评判均为 `gpt-5.4-mini`(经 OpenAI 兼容网关)。
- **单次检索召回失败是确定性的**(纯检索、无 LLM)，下面列出的 case 可精确复现。注意 `run_eval` 现在报**两个**检索指标:`检索召回`(对整条复合问题单次检索)与 `源覆盖`(agent 实际、含编排拆子问题逐个检索)。本节的「漏检」指**单次检索**口径;agent 端到端覆盖要高得多(多跳 0.70 → 0.83)。
- **答案 / 引用 / 拒答类失败依赖 LLM,输出有随机性**——下面的样本来自对 14 条拒答题的一次重跑,用来**例证失败模式**,具体哪几条命中会因 run 而异(聚合计数见 README:拒答 13/14、out_of_scope 意图 0.80;幻觉引用不同 run 在 0–4 间波动,均落在该拒答的题上)。

复现：`python -m citelocal_agent.eval.run_eval --split offline_sample`(召回部分逐条打印,确定性)。

---

## 1. 单次检索召回失败(确定性,引入双阈值门控前 35 / 145 条 recall < 1.0,现 27 条)

「单次检索召回」在 `TOP_K=4` 下计算,口径是「一题所需的**每一段**证据都要命中」(严格 all-of)。**注意:这衡量的是对整条复合问题做一次检索;agent 实际走编排拆子问题,端到端覆盖高得多(多跳 0.80 / 总体 0.95)。** 下面 1a–1c 的清单是**引入双阈值门控前**(单阈值 `SCORE_THRESHOLD=0.0`)的 35 条漏检;1d 的诊断 + 双阈值门控落地后,浅负簇被救回、单次检索漏检降到 **27 条**(多跳 22 / 单源 2 / 定义 3),多跳单次召回 0.70 → 0.80。下面清单保留作根因记录。

### 1a. 多跳题「漏掉第二篇」(recall 0.5,主导模式)

两篇必需文档只命中一篇。被漏掉的几乎总是**更宽泛的「枢纽型」笔记**(问题措辞偏向另一篇)。漏检最频繁的文档:

| 被漏掉的源文档 | 漏检次数 | 例子 case id |
|---|---|---|
| `transformers-and-attention.md` | 5 | `os_attention_vs_bm25`, `os_ann_dense_embeddings`, `os_mh_contextual_transformer`, `os_mh_pretrain_finetune_encoder` |
| `dense-and-sparse-retrieval.md` | 5 | `os_recall_ceiling_retrieval`, `os_mh_agentic_rag_retrieval`, `os_mh_eval_recall_hybrid` |
| `quantization.md` | 3 | `os_mh_qlora_quantization`, `os_mh_distill_vs_quant` |
| `evaluating-rag-systems.md` | 3 | `os_mh_tool_call_citations`, `os_mh_routing_agents` |
| `tokenization-and-embeddings.md` | 3 | `os_mh_embedding_dim_ann`, `os_mh_vectordb_stores_embeddings`, `os_mh_icl_token_cost` |

> 例:`os_mh_qlora_quantization`「QLoRA 如何把量化和低秩适配结合」——命中了 `qlora.md`,漏了 `quantization.md`(reranker 对整条问题给 `quantization.md` 的最佳 chunk 打了 **-5.36** 分,低于 0.0 阈值被滤掉;详见 1d)。

### 1b. 单源题彻底漏检(recall 0.0,5 条)

该题唯一该命中的文档没进 top-4——reranker 对整条问题给它的最佳 chunk 打了负分、被阈值滤掉(见 1d)。是 README 吹「100+ 干扰文档仍稳召回」**翻车的反例**:

| case id | 类别 | 该命中却漏掉的文档 | 问题 |
|---|---|---|---|
| `os_rag_pipeline_steps` | single_paper | `retrieval-augmented-generation.md` | RAG 流水线主要阶段 |
| `os_per_category_reporting` | single_paper | `evaluating-rag-systems.md` | 为何按类别而非只看总体报指标 |
| `os_ft_fine_tuning` | definitional | `fine-tuning-and-transfer-learning.md` | 什么是微调、主要代价 |
| `os_quantization` | definitional | `quantization.md` | 什么是量化、权衡什么 |
| `os_ndcg` | definitional | `ndcg.md` | nDCG 比简单指标多算了什么 |

### 1c. 多跳题两篇全漏(recall 0.0,2 条)

`os_mh_lora_transformer`(漏 `fine-tuning-and-transfer-learning.md` + `transformers-and-attention.md`)、`os_mh_react_tool_calling`(漏 `react-reason-and-act.md` + `tool-calling-mechanics.md`)。

### 1d. 根因诊断(确定性探查,改写了最初的猜测)

最初猜「第二篇源在候选池里、被同源 chunk 挤出 top-4」,**实测证伪**。逐层探查 35 条漏检:

- **候选池里都有**:把 `candidate_k` 拉到 200、关掉阈值,35 条漏掉的源**全部能检索到**(深度漏检 = 0)。所以不是「没召回到」。
- **真因 = reranker 负分被阈值砍**:这些源的最佳 chunk 经 cross-encoder 重排后分数 **< 0.0 阈值**,被 [retriever.py](../src/citelocal_agent/retriever.py) 的相关性阈值滤掉。分两簇:**浅负(0 ~ -3)**就差一点(如 `mean-reciprocal-rank.md` -0.05、`react-reason-and-act.md` -0.08、`quantization.md` -1.13);**深负(-6 ~ -11)**则是 reranker 真的不认为该文档与问题措辞相关。
- **「来源多样性 cap」是死路(负面结论,如实记录)**:据上面那个错误猜测实现了「每来源限流 + 回填」的 top-k 选取,在 cap∈{2,3} 下对全部 159 条重算召回——**与无 cap 逐位相同,零提升**。因为漏掉的源根本不在过阈值候选里,没有可「让位」的对象。该改动已撤回,不留死代码。
- **全局下调阈值能救召回但伤拒答**:阈值 0.0 → -2.0 时总体召回 0.86→0.91、多跳 0.70→0.78;但「该拒答」题的检索泄漏从 2/14 升到 4/14、-3.0 升到 6/14,直接威胁拒答 / 幻觉(本就是弱项)。原校准注释「[-2.5,2.5] 拒答仍 100%」在 106 篇规模下已不成立。

### 修法落地:双阈值门控(已合入)

按上面诊断,把阈值的两个职责拆开([retriever.py](../src/citelocal_agent/retriever.py) `search()`):
- `SCORE_THRESHOLD`(默认 0.0)仍是**拒答闸门**——最高分不过它就返回空,拒答行为逐字节不变。
- 新增 `SUPPORT_THRESHOLD`(默认 -2.5)是**闸门开后**才用的更松的纳入线,把浅负簇的第二源 chunk 收进来。
- 护栏:`support = min(support, gate)`,绝不严于闸门;设为等于闸门即关闭。

**实测(确定性 + 全量 LLM):** 单次多跳召回 0.70 → **0.80**、总体 0.855 → 0.891、端到端覆盖 0.83 → **0.87**,而**拒答泄漏在所有次阈值下都停在 2/14**(对比全局下调到 -2.0 是 4/14)、全量 LLM 拒答准确率 13/14 不变。单源 / 定义不回退。即:拿到召回增益、规避了拒答代价。

### 仍待改进(按性价比)

- **reranker 是剩余真瓶颈**:27 条仍漏的多为**深负簇**(reranker 给第二源 -6 ~ -11),双阈值救不动。已做[reranker 选型实验](reranker-selection.md)(`scripts/rerank_bakeoff.py`):`bge-reranker-base` 把多跳排序天花板 0.843→0.873、拒答 AUC 还更好,是当前最佳候选(待重校准阈值 + 全量验收后定型);增益温和。另一条路是让重排只面对编排拆出的**单跳子问题**而非复合问题。
- 1b 单源漏检:查询改写 / 关键实体加权,把该文档 chunk 在重排里抬过闸门。
- **度量改进已落地**:`run_eval` 同时报「单次检索召回」(检索子系统诊断)与「agent 端到端源覆盖」(系统真实表现,多跳 0.87)。

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
