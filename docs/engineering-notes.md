# 工程踩坑记录（Problems & Solutions）

记录 docagent 在开发 / 接入真实模型 / 评测过程中遇到的问题与解决方案,方便复用与避坑。

---

## 1. 强制 `tool_choice` 与模型「思考模式」冲突

**现象**:接入某些 OpenAI 兼容端点后,ReAct 检索循环(`bind_tools(..., tool_choice="any")`)和结构化输出(`with_structured_output`)报
`The tool_choice parameter does not support being set to required or object in thinking mode`。

**原因**:部分模型默认开启 reasoning/thinking 模式,该模式下不接受强制工具调用(`tool_choice=required` 或指定具体工具)。而我们的设计依赖强制工具调用来保证「必定检索 / 必定产出带引用的 `Answer`」。

**解决**:加一个**通用逃生口** `LLM_EXTRA_BODY`(JSON,经 `configuration.llm_call_kwargs()` 作为 `extra_body` 透传给 OpenAI 客户端),可在不改代码的前提下传 provider 专属请求字段。默认空,标准 OpenAI 不受影响。本地按需设置,例如关闭思考模式。

**教训**:不要把「强制工具调用」这种平台相关行为写死;留一个 env 级的请求体逃生口,适配各种 OpenAI 兼容网关。

---

## 2. 环境变量没有按类型转换

**现象**:设了 `RECURSION_LIMIT=20` 后,图执行报 `'<' not supported between instances of 'str' and 'int'`。

**原因**:`Configuration.from_runnable_config` 直接把 `os.environ` 的字符串值塞进 dataclass,`recursion_limit` 成了字符串 `"20"`,传进 LangGraph 后 `step < limit` 比较崩溃。`TOP_K`、`SCORE_THRESHOLD` 等所有数值型 env 覆盖都有同样隐患。

**解决**:`from_runnable_config` 按字段默认值的类型对 env 值做强制转换(int / float / bool / dict)。

**教训**:任何「env 覆盖配置」的入口都要做类型转换,别假设 env 是目标类型。

---

## 3. 后台 / 无头任务里 HuggingFace 的 http client 被关闭

**现象**:在后台任务里跑 `run_eval` / pytest,模型加载阶段 `hf_hub_download` 抛 `RuntimeError: Cannot send a request, as the client has been closed.`;同样的脚本在前台直接跑却正常。

**原因**:后台任务的执行环境会关闭进程继承的部分 http 资源,导致 `huggingface_hub` 的全局 client 在「加载时联网校验 etag」这一步失效。

**解决**:模型已在本地缓存的前提下,跑命令时加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`,加载完全走缓存、零网络调用,彻底规避。

**教训**:CI / 批处理 / 无头截图等场景,凡是模型已缓存,优先开离线模式,既稳又快。

---

## 4. 结构化输出偶发不合 schema,拖垮整条复杂路径

**现象**:复杂问题走 orchestrator 时,约 10% 的调用里 planner 的结构化输出缺字段(如 `reasoning` Field required),抛 `ValidationError`,整个请求 500;`run_eval` 也因一例异常而整轮中断。

**原因**:不同模型对结构化输出的遵从度不一,弱一些的模型偶发吐坏 JSON / 缺字段。

**解决**:**双层兜底**——
- planner 内层 `try/except`:结构化输出失败时回退为「把原问题当作单个子问题」;
- `orchestrator_node` 外层 `try/except`:复杂路径任何环节失败(坏结构化输出、researcher 递归预算耗尽……)都降级到简单检索循环,请求绝不崩。
- `run_eval` 逐例 `try/except` + 统计 errored 数,单例失败不终止整轮评测。

**教训**:依赖 LLM 结构化输出的环节必须假设它会偶尔失败,给确定性的降级路径。

---

## 5. pytest 跨用例复用被缓存的 client 导致泄漏

**现象**:LLM 端到端测试里,第二个用例起报 `client has been closed`,甚至先前能过的用例也跟着失败。

**原因**:`get_default_agent()` / `get_chat_agent()` 是 `lru_cache` 单例,其 http client 被某个模块的 pytest teardown 关闭后,跨测试模块复用到了已关闭的 client。

**解决**:gated 测试的 `agent` fixture 改为每个用例用 `build_agent(...)` 新建,隔离 client 生命周期。

**教训**:进程内单例(尤其持有网络连接的)在测试里要警惕跨用例 / 跨模块复用;测试用新实例最稳。

---

## 6. 多轮记忆在浏览器里「没生效」

**现象**:服务端做了 checkpointer 多轮记忆,但 Web UI 里追问得不到上下文。

**原因**:静态前端每次 `POST /api/ask` 都不带 `session_id`,服务端每次都开新线程,记忆自然不连续。

**解决**:页面加载时生成一个 `session_id`,每次请求都带上;服务端按 `thread_id` 关联多轮。

**教训**:有状态能力要端到端打通——后端支持不等于前端用上了,记得在调用方传会话标识。

---

## 7. 子图递归预算的继承问题

**现象**:想给「简单路径」和「复杂 orchestrator 路径」设不同的步数上限,但把已编译子图直接 `add_node` 进父图时,两条路径都继承了同一个顶层 `recursion_limit`。

**解决**:用**包装节点**分别 `invoke` 各自的内层图,并显式传该路径自己的 `recursion_limit`;顶层图只剩「路由 → 单节点」,预算全部进 `Configuration`,调用方不再硬编码。

---

## 8. 持久稀疏索引(bm25s)的两个注意点

- **分词一致性**:build 索引与 query 必须用同一分词配置,否则召回异常。
- **阈值复核**:更换稀疏检索后端 / 重排模型后,要用 `scripts/calibrate_threshold.py` 重新校准相关性阈值(它决定「不在文档里就拒答」的边界)。
- 索引落盘到 `{chroma_path}/bm25s/{collection}/`,query 时 `mmap` 加载,启动不再把全语料读进内存。

---

## 9. 演示截图要「真」

**教训**:不要用硬编码的 mock 数据冒充演示图。`docs/ui-answer.png` 是用 Playwright 驱动**真实运行的应用**跑一段真实多轮对话后截下来的——检索、作答、引用、会话记忆都是真的(`?demo` 仅作离线无 key 预览之用)。
