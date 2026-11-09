# 中文 RAG 混合检索与文档问答

基于 [mohammadyakub-ai/rag-pipeline-hybrid-search](https://github.com/mohammadyakub-ai/rag-pipeline-hybrid-search) 的中文实践项目，提供混合检索、中文重排、带引用问答、裁判校验、拒答和执行追踪。保留上游提交历史及 MIT 许可证。

## 已实现的功能

- Markdown、文本、HTML、PDF 加载与固定、递归、语义切分。
- 本地向量模型与 ChromaDB，结合支持中文分词的 BM25。
- 向量与关键词各召回 60 段，通过 RRF 融合，重排后最多使用 20 段、12000 字符上下文。
- 支持中文的多语言 Cross-Encoder 重排，可通过 `RAG_RERANKER_MODEL` 选择模型。
- DeepSeek、OpenRouter 或 OpenAI 兼容聊天服务，生成带 `[N]` 引用的中文答案。
- 独立裁判提示及严格 JSON 解析，校验引用、完整度与置信度后决定回答或拒答。
- FastAPI 接口、Streamlit 中文界面及 OTLP/HTTP 执行追踪。
- 20 题历史回归集，以及「rag基线」50 题集合，后者包含 40 道可答题、10 道自拟应拒答题和 68 篇配套原文。

## 安装与启动

需要 Python 3.12。以下命令在 Windows PowerShell 中执行。

```powershell
git clone https://github.com/dream001871-cell/rag-pipeline-hybrid-search.git
cd rag-pipeline-hybrid-search
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

在 `.env` 填写 `DEEPSEEK_API_KEY` 可启用真实问答。向量和重排在本机运行，首次启动需要下载模型。也可以配置 OpenRouter 或 OpenAI 服务，详见 [本地运行说明](LOCAL_RUN.md)。

```powershell
powershell -ExecutionPolicy Bypass -File .\start-local.ps1
.\.venv\Scripts\python.exe scripts/seed.py --api http://127.0.0.1:8000 --timeout 600
```

- 中文界面位于 http://127.0.0.1:8501。
- API 文档位于 http://127.0.0.1:8000/docs。
- 健康检查位于 http://127.0.0.1:8000/healthz。

纯检索使用 `POST /v1/retrieve`。文档上传使用 `POST /v1/ingest`。问答使用 `POST /v1/ask`。

```json
{"question": "What is FastAPI?", "top_k": 20, "retrieval_mode": "hybrid"}
```

停止本机服务可运行 `powershell -ExecutionPolicy Bypass -File .\stop-local.ps1`。

## 执行追踪

设置 `RAG_TRACE_ENDPOINT` 为 OTLP/HTTP 接收地址。例如配套本地 Evals.si 服务使用 `http://127.0.0.1:8088/v1/traces`。测试中台需要单独部署，本项目提供调用与上报适配代码。

问答响应头 `X-Trace-Id` 对应检索、生成、引用校验和最终判定各阶段。日常服务以 `rag-api-production` 上报阶段耗时、模型名和结果状态，正文留在服务端的问答响应中。追踪接收端故障会记录日志，问答接口继续返回结果。

## 评测记录

固定小型资料范围内的历史模型复评结果如下。生成及评审均使用 DeepSeek `deepseek-flash`，语义复评采用独立提示。每题单次生成，结果受模型判断影响，尚未人工逐题验收。

| 集合与方法 | 最终回答 | 复评正确 |
| --- | --- | --- |
| 20 题原基线 | 2/20 | 2/20 |
| 20 题修复裁判提示与解析后 | 13/20 | 12/20 |
| 20 题增加中文重排后 | 16/20 | 16/20 |

「rag基线」50 题运行记录于 2026-11-07。

| 指标 | 结果 |
| --- | --- |
| 可答题语义正确 | 32/40（80%） |
| 可答题误拒答 | 8/40（20%） |
| 应拒答题正确拒答 | 10/10（100%） |
| 总体语义正确 | 42/50（84%） |
| 平衡准确率 | 90% |
| 执行追踪核对 | 50/50 |

题目和语料来自小型选集，配套语料规模与官方完整基准不同。这些数值用于本项目的诊断，不能直接与完整 CRUD-RAG 分数比较。20 题参与了调优，50 题中的拒答题为本地构造。参考答案和标签只供评审使用，建库时只导入对应 `corpus/` 中的原文。

## 代码入口

| 文件 | 作用 |
| --- | --- |
| `src/rag_pipeline/api.py` | 组装检索、生成、校验与接口 |
| `src/rag_pipeline/generation.py` | 上下文预算、普通问答和裁判消息 |
| `src/rag_pipeline/fusion.py` | 双路候选与 RRF 融合 |
| `src/rag_pipeline/reranker.py` | 重排模型与选择策略 |
| `src/rag_pipeline/citation.py` | 引用解析与证据判定 |
| `src/rag_pipeline/confidence.py` | 完整度判分与故障信息 |
| `src/rag_pipeline/unknown.py` | 回答或拒答门禁 |
| `src/rag_pipeline/tracing.py` | 阶段事件和 OTLP 上报 |
| `data/benchmarks/current_dataset.json` | 当前 50 题集合入口 |

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py tests/test_judge_protocol.py tests/test_answer_gates.py tests/test_generation.py tests/test_citation.py tests/test_confidence.py tests/test_retrieval_plan.py tests/test_structured_citations.py tests/test_bm25_chinese.py tests/test_deepseek.py -q
```

公开前上述 124 项相关测试通过。测试使用隔离组件，实际模型评测单独执行。索引、模型缓存、密钥及本机详细调用记录在运行后生成。

## 来源和许可证

代码沿用上游 [MIT 许可证](LICENSE)，保留 `Copyright (c) 2026 Yakub`。中文化、中文检索、裁判校验改进和执行追踪在上游代码基础上实现。

中文题目来源为 [IAAR-Shanghai/CRUD_RAG](https://github.com/IAAR-Shanghai/CRUD_RAG)，题目出处、原始编号及选题记录保存在各集合目录。10 道应拒答题为本地自拟，标签适用于固定资料范围。所附新闻及第三方 FastAPI 文档的权利归各自权利人，代码 MIT 许可证不改变这些第三方材料的权利。上游原始说明保存在 [README.en.md](README.en.md)。
