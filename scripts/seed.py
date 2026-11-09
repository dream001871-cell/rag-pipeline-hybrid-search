"""Seed the RAG API with a sample documentation corpus.

Uploads every supported file found under --corpus-dir to the API's POST
/v1/ingest endpoint so reviewers can test the full stack immediately.

Idempotent: re-running against an already-seeded API simply reports each file
as already indexed (duplicates > 0).

Usage:
    python scripts/seed.py --api http://localhost:8000 --corpus-dir data/raw/fastapi_docs
    python scripts/seed.py --dry-run --corpus-dir /app/corpus
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import mimetypes
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rag_pipeline.embedding import describe_embedding_service

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}


def discover(corpus_dir: str, explicit_files=None) -> list:
    """Return supported corpus files. Explicit files override discovery."""
    if explicit_files:
        return [Path(f) for f in explicit_files]
    root = Path(corpus_dir)
    if not root.is_dir():
        return []
    return [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def seed_file(
    api_base: str, path: Path, corpus_dir: str | None = None, timeout: int = 120
) -> requests.Response:
    """Upload one file; the multipart filename is relative to the corpus
    root (falling back to the basename for explicit file paths) so the API
    stores a meaningful ``source_file`` instead of a temp path."""
    try:
        name = str(path.relative_to(Path(corpus_dir))) if corpus_dir else str(path)
    except ValueError:
        name = path.name
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    with open(path, "rb") as fh:
        content = fh.read()
    return requests.post(
        f"{api_base}/v1/ingest",
        files={"file": (name, content, mime)},
        timeout=timeout,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="将样例文档导入 RAG 服务。"
    )
    parser.add_argument(
        "--api",
        default=os.environ.get("API_BASE_URL", "http://localhost:8000"),
        help="后端地址（默认使用 API_BASE_URL 或 http://localhost:8000）",
    )
    parser.add_argument(
        "--corpus-dir",
        default="data/raw/fastapi_docs",
        help="扫描文档的目录（默认：data/raw/fastapi_docs）",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="指定文件路径，替代目录扫描",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="仅列出文件，不执行上传")
    parser.add_argument("--timeout", type=int, default=120,
                        help="单个文件上传的等待时间，单位为秒（默认：120）")
    parser.add_argument("--quiet", action="store_true",
                        help="仅输出失败信息和最终汇总")
    parser.add_argument("--reset", action="store_true",
                        help="导入前清空向量与关键词索引")
    args = parser.parse_args(argv)

    if args.reset and args.dry_run:
        parser.error("--reset 与 --dry-run 不能同时使用")

    if args.reset:
        try:
            resp = requests.post(f"{args.api}/v1/reset", timeout=args.timeout)
        except requests.RequestException as exc:
            print(f"清空索引失败： {exc}", file=sys.stderr)
            return 1
        if resp.status_code != 200:
            print(
                f"清空索引失败： HTTP {resp.status_code} {resp.text[:200]}",
                file=sys.stderr,
            )
            return 1
        if not args.quiet:
            print("向量与关键词索引已清空。")

    sharp = not args.quiet
    if sharp:
        print(f"向量模型： {describe_embedding_service()}")

    files = discover(args.corpus_dir, args.files)
    if not files:
        print(f"在 '{args.corpus_dir}' 中没有找到支持的文档。", file=sys.stderr)
        return 2

    if args.dry_run:
        for f in files:
            print(f"- {f}")
        print(f"[预览] 将向 {args.api} 导入 {len(files)} 个文件")
        return 0

    if not args.quiet:
        print(f"在 {args.corpus_dir}/ 找到 {len(files)} 份待索引文档")
        print(f"正在向 {args.api} 导入 {len(files)} 个文件……")

    failed = 0
    seeded_doc_ids = []
    for path in files:
        try:
            resp = seed_file(args.api, path, corpus_dir=args.corpus_dir, timeout=args.timeout)
            if resp.status_code == 201:
                data = resp.json()
                seeded_doc_ids.append(data.get("doc_id"))
                print(
                    f"成功 {path.name}：新增 {data.get('chunks_added_dense', 0)} 个片段，"
                    f"{data.get('duplicates', 0)} 个重复片段，文档编号={data.get('doc_id')}"
                )
            else:
                failed += 1
                print(f"失败 {path.name}： HTTP {resp.status_code} {resp.text[:200]}")
        except requests.RequestException as exc:
            failed += 1
            print(f"失败 {path.name}： {exc}")

    if failed:
        print(f"导入结束：{len(files) - failed}/{len(files)} 个文件成功。")
        return 1

    print(f"导入完成：全部 {len(files)} 个文件已处理。文档编号：{', '.join(seeded_doc_ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())