# Reranker 选型实验

把「换哪个 cross-encoder reranker」当成一项可复现、可比较的工程决策来做,而不是凭感觉换模型。结论:**bge-reranker-base 是当前最佳候选,但尚未定型**(见末尾)。

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

## 推荐与「定型」前置条件(决策待定)

**候选:`BAAI/bge-reranker-base`**——质量/拒答/延迟综合最优;若追求极致多跳且能忍 3× 延迟,再考虑 `bge-reranker-v2-m3`。

**尚未定型。** 真正切换默认 `RERANKER_MODEL` 前必须:

1. **重校准阈值**:bge 输出 ≈[0,1] sigmoid,而非 ms-marco 的宽 logit。`SCORE_THRESHOLD`(拒答闸门)需重跑 [`scripts/calibrate_threshold.py`](../scripts/calibrate_threshold.py),`SUPPORT_THRESHOLD` 需重新 sweep。
2. **全量 LLM 验收**:`run_eval --split offline_sample` 确认多跳单次召回/端到端覆盖上升,且**拒答准确率、幻觉引用不恶化**。
3. **权衡 CPU 延迟**:bge-base 每次检索 ~377ms(现状 ~104ms),交互式与批量评估都会变慢——按部署环境(有无 GPU)再定。

> 本实验只测**排序质量**;头牌的单次召回是检索子系统诊断指标,agent 端到端覆盖(已 0.87)才是系统真实表现。换 reranker 同时抬两者,但幅度温和。
