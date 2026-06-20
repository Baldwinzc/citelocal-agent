# Reranker 选型实验

把「换哪个 cross-encoder reranker」当成一项可复现、可比较的工程决策来做,而不是凭感觉换模型。**结论:已定型 `bge-reranker-v2-m3`**(按「结果优先」取排序最强档;延迟列为后期优化项。详见末尾)。

## 背景

多跳是检索最弱的一环。诊断(见[失败案例库](failure-cases.md) §1d)表明:仍漏的黄金源**能被检索到**,但当前 reranker `ms-marco-MiniLM-L-6-v2` 对整条复合问题给它打负分。阈值调优(双阈值门控)已把多跳单次召回抬到 0.80、接近**关掉阈值时的排序天花板 0.843**——再往上只能靠**更强的 reranker 把第二源排进 top-k**。本实验量化「换强 reranker 到底值多少」。

## 方法学(为何可比)

脚本:[`scripts/rerank_bakeoff.py`](../scripts/rerank_bakeoff.py),确定性、零 LLM。

- **候选池固定**:dense+BM25+RRF 的候选池不依赖 reranker,各模型共用同一组候选,**只换重排打分**——隔离出纯排序质量。
- **recall@k 关掉阈值**:衡量「黄金源是否被排进 top-k」,与绝对分数尺度无关,**跨模型可比**(各模型 logit/sigmoid 尺度不同)。
- **拒答 AUC**:top-1 分数区分「可答(in_scope)」vs「该拒答(out_of_scope/no_answer)」的 Mann-Whitney AUC,1.0 = 完美可分——衡量换模型后**还能否校准出拒答阈值**(abstention headroom)。
- **延迟**:本机(CPU、无 GPU)每次 `search()` 的平均墙钟,毫秒/题——速度是真实决策项。
- 语料:`offline_sample`(106 篇 `sample_notes/`,159 题,`candidate_k=20`)。复现:`python scripts/rerank_bakeoff.py --split offline_sample`。

## 结果

| 模型 | 体量 | recall@4 | recall@10 | multi_hop@4 | 拒答 AUC | ms/题(CPU) |
|---|---|---|---|---|---|---|
| `ms-marco-MiniLM-L-6-v2`(现默认) | ~22M | 0.941 | 0.979 | 0.843 | 0.951 | **104** |
| `ms-marco-MiniLM-L-12-v2` | ~33M | 0.939 | 0.976 | 0.836 | 0.947 | 150 |
| **`BAAI/bge-reranker-base`** | ~278M | 0.953 | **0.986** | 0.873 | **0.972** | 377 |
| `BAAI/bge-reranker-v2-m3` | ~568M | **0.959** | 0.983 | **0.889** | 0.971 | 1144 |
| `mixedbread-ai/mxbai-rerank-base-v1` | ~184M | 0.940 | 0.983 | 0.840 | 0.957 | 524 |

(recall 关阈值 = 纯排序;multi_hop 是最该改善的类别;单源/定义各模型都已 ≈1.00,无空间。)

## 分析

- **`ms-marco-L-12` 与 `mxbai-base` 被全面支配**:排序不优于(甚至略低于)现状的 L-6,延迟却更高——排除。「廉价加层」(L-6→L-12)无效。
- **`bge-reranker-base` 是性价比甜点**:multi_hop 0.843→**0.873**(+0.03)、recall@10 与拒答 AUC **全场最佳**,延迟为 v2-m3 的 1/3(377 vs 1144 ms)。
- **`bge-reranker-v2-m3` 排序最高**(multi_hop 0.889,+0.046),但比 bge-base 多花 **3× 延迟**只换 +0.016,CPU 上边际递减;recall@10 反而略低于 base。
- **拒答不受损**:bge 两款 AUC(0.97)略优于现状(0.95),换模型后仍能校准出干净阈值。
- **增益温和且只在多跳**:单源/定义已满,提升空间全在多跳的 +0.03~0.05。叠加阈值过滤与「agent 已靠编排拆解把端到端覆盖做到 0.87」,落到头牌指标的实际收益会更小。

## 定型:`BAAI/bge-reranker-v2-m3`(已采纳)

决策口径为「结果优先,延迟后期再优化」,故取排序最强的 **`bge-reranker-v2-m3`**(而非性价比档 bge-base)。已落地为默认 `RERANKER_MODEL`,完成两步定型前置:

1. **重校准阈值(已完成)**:bge 输出 ≈[0,1] sigmoid。用 [`scripts/calibrate_threshold.py`](../scripts/calibrate_threshold.py)(已 generalize 成按分数自动定 sweep 范围)在 offline_sample 上重校:in_scope top 分 ~0.14–0.99、refuse ~0.00–0.05(两个 topically-near 的 no_answer 离群 0.52/0.86)。
   - `SCORE_THRESHOLD = 0.2`(闸门):recall ~0.99、abstain ~0.86(拒答泄漏 2/14,与旧模型持平)。脚本按 `f1+abstain` 的 argmax 是 ~0.525,但那会错误拒答 ~10% 可答题,**故人工选 0.2 保 recall**。
   - `SUPPORT_THRESHOLD = 0.0`:闸门开后纳入任意 >0 分 chunk;sweep 显示多跳单次召回随 support 降而升,0.0 时达 0.889,**泄漏全程 2/14 不变**。
2. **全量 LLM 验收(已完成,159 条 / gpt-5.4-mini)**:相对换前(ms-marco)——单次召回 0.89→**0.94**、端到端覆盖 0.95→**0.97**、多跳召回 0.80→**0.89**、多跳覆盖 0.87→**0.92**、答案 97→**99%**,**拒答 13/14 不变、幻觉 1**(在带内)。单源/定义召回升到 0.98。
   - **代价/观察**:引用接地总体 93→92(噪声级);definitional 0.91→0.84、numeric 0.71(n=7,波动大)。疑因 `SUPPORT_THRESHOLD=0` 较激进、纳入弱 chunk 稀释引用——若在意可上调 support 到 ~0.05(多跳 0.889→0.861 的小幅让步)。

### 延迟(后期优化项,已知)

bge-v2-m3 在本机 CPU ~1144ms/检索(ms-marco ~104ms),交互与批量评估都更慢。按「结果优先」暂接受;后期优化方向:GPU 推理、换 bge-base(~377ms,质量略降)、或两段式重排。**单测已把 reranker 钉在小模型**,故 CI 不受拖累(测的是检索逻辑,非模型质量)。

> 本实验只测**排序质量**;头牌的单次召回是检索子系统诊断指标,agent 端到端覆盖才是系统真实表现。复现:`python scripts/rerank_bakeoff.py --split offline_sample`。
