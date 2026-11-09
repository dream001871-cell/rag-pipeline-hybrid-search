# 当前检索与问答方案

2026-11-05 采用扩大单次检索的 B 方案，作为正式默认流程。
这里的 B 指最近三组生成对照中的扩大候选方案。

## 执行顺序

1. 使用原问题分别进行向量检索和 BM25 检索，每路最多召回 60 段。
2. 使用 RRF 融合，常数 60，向量与关键词权重分别为 0.7、0.3。
3. 对融合结果前 60 段进行 Cross-Encoder 重排，默认取前 20 段。
4. 按排名选择总正文不超过 12,000 字符的完整片段，不截断片段。放不下的片段跳过，后续较短片段仍可加入。
5. 为选定片段重新连续编号，生成中文回答，默认最多输出 1536 token。
6. 使用相同片段校验引用、计算置信度和检查最低完整度，然后回答或拒答。

接口请求中的 `top_k`、`retrieval_mode`、`mode` 字段继续兼容；省略时默认混合检索和前 20 段。
工作台默认也是 20 段。纯向量、纯关键词模式保留用于原有诊断功能。
没有查询改写、补充检索、父章节扩展或 Agent 循环。

生产索引、语料和数据格式继续使用现有版本，不需要重新生成向量。
本地模型缺失时仍沿用项目原有的重排降级机制；降级运行不能当成真实 Cross-Encoder 效果。

## 2026-11-06 中文测试改进

本机已修复裁判提示包装和 JSON 判分解析，并启用支持中文的 `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` 重排。向量模型、两路召回量、RRF 权重、上下文预算以及两个拒答阈值保持不变。

重排通过 `RAG_RERANKER_MODEL` 配置模型仓库名或本地目录，显式传入模型参数时优先使用该参数；未配置时默认使用支持中文的 `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`。本机 `.env` 指向已下载并校验的 `.cache/models/mmarco-mMiniLMv2-L12-H384-v1`。旧英文模型的本地缓存已移除，历史评测记录保留原模型信息。

固定 20 道中文题的三轮结果分别为：最终回答 2、13、16 题；同模型独立提示复评判为正确 2、12、16 题。后两轮没有裁判协议错误。最终结果是小型调优集的模型复评，不能替代完整基准或人工验收。详见 [三轮对照报告](data/benchmarks/crud_rag_20/evaluation/improvement_report.md)。

## 证据与限制

最近 8 题诊断子集的单次对照结果：A 为 4/8，B 和 C 各为 6/8。
B 每题平均 3.35 秒，C 为 5.94 秒，C 的总 token 用量为 B 的 2.31 倍。
这一结果支持先采用 B，不能证明其在全部问题上的准确率为 75%。

该实验使用独立的无去重索引，原平台索引含已有去重结果；正式流程还增加了引用与完整度门禁。
因此正式平台的最终通过率需要单独测量，不能直接套用门禁前的小样本结果。
q004 还存在首页示例与入门教程示例的题干歧义，历史结果保持原样。

历史证据保存在 `.local/agentic-abc/20261105-181119-157991/` 的
`report.md`、`manifest.json`、`answers.jsonl`、`review.json` 和 `summary.json`。
更早的原文/父章节对照结果仍保存在 `.local/context-comparison/` 与 `.local/context-answer-subset/`。
这些目录仅作为历史数据，不再提供 A/C 执行入口；历史文档中的旧脚本命令已失效。
8 题选择和评分要点仍保存在 `data/context_answer_subset.json`，原有 52 题数据不变。

## 验证与运行

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_retrieval_plan.py tests/test_fusion.py tests/test_api.py tests/test_generation.py tests/test_answer_gates.py tests/test_eval_metrics.py tests/test_chunking_comparison.py tests/test_dashboard.py -q
powershell -ExecutionPolicy Bypass -File .\stop-local.ps1
powershell -ExecutionPolicy Bypass -File .\start-local.ps1
.\.venv\Scripts\python.exe scripts/verify_local.py
```

测试使用隔离组件，不请求真实模型 API。HTTP 验证读取已有索引，检查默认 20 段及真实重排结果。
