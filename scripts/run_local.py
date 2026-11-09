"""使用项目自己的配置启动本机服务。"""

import argparse
import os
from pathlib import Path
import runpy
import sys

from dotenv import load_dotenv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动本地 API 或检索界面")
    parser.add_argument("service", choices=["api", "dashboard"])
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    os.chdir(project_dir)
    sys.path.insert(0, str(project_dir / "src"))

    # 仅此本机入口优先使用 .env，避免继承系统里失效的密钥。
    load_dotenv(project_dir / ".env", override=True)
    # 填好密钥并重启后自动进入问答模式；显式设置 true/false 时保留用户选择。
    if os.environ.get("RAG_RETRIEVAL_ONLY", "auto").lower() == "auto":
        has_chat_key = any(os.environ.get(name, "").strip() for name in (
            "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"
        ))
        os.environ["RAG_RETRIEVAL_ONLY"] = "false" if has_chat_key else "true"
    if not os.environ.get("OPENAI_BASE_URL", "").strip():
        os.environ.pop("OPENAI_BASE_URL", None)
    os.environ.setdefault("HF_HOME", str(project_dir / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUTF8", "1")

    if args.service == "api":
        import uvicorn

        uvicorn.run("rag_pipeline.api:create_app", factory=True, host="127.0.0.1", port=8000)
    else:
        sys.argv = [
            "streamlit", "run", str(project_dir / "dashboards" / "query_dashboard.py"),
            "--server.address=127.0.0.1", "--server.port=8501",
            "--server.headless=true", "--browser.gatherUsageStats=false",
        ]
        runpy.run_module("streamlit", run_name="__main__")
