# 本地运行说明

使用 Python 3.12，按仓库 README 创建虚拟环境并安装 `requirements.txt`。`requirements-local.lock.txt` 记录开发时的依赖版本，供排查环境差异。

将 `.env.example` 复制为 `.env`，填写实际模型密钥。已有本地模型时可将 `RAG_RERANKER_MODEL` 设为模型目录，或使用默认多语言模型仓库名让程序下载。

```powershell
powershell -ExecutionPolicy Bypass -File .\start-local.ps1
.\.venv\Scripts\python.exe scripts/seed.py --api http://127.0.0.1:8000 --timeout 600
```

本机 API 使用 8000 端口，中文界面使用 8501 端口。日志保存在 `.local/`，向量与关键词索引保存在 `data/chroma/` 和 `data/bm25/`。

导入中文测试语料时应为不同语料准备独立索引。50 题的配置入口为 `data/benchmarks/current_dataset.json`，只将其 `corpus` 指向的原文用于检索。参考答案、标签和评审记录用作独立评测资料。

`start-local.ps1` 默认创建 FastAPI 样例索引所在的服务。历史 20 题评测脚本具有其对应的数据入口，运行前应核对各脚本参数，50 题入口由 `--variant holdout --batch <批次标识>` 指定。

```powershell
.\.venv\Scripts\python.exe scripts/run_crud_eval_service.py --variant holdout --batch baseline-local
```

上述评测服务使用 8001 端口，并为当前资料集创建独立索引。它要求 `RAG_TRACE_ENDPOINT` 指向已部署的中台 OTLP 接收地址。跨 WSL 连接可使用 `scripts/run_crud_eval_relay.py`，参数中的虚拟网卡和来源地址需根据当前环境设置。

更新 `.env` 配置后重启服务加载。Windows 中可执行以下停止命令后重新启动。

```powershell
powershell -ExecutionPolicy Bypass -File .\stop-local.ps1
```
