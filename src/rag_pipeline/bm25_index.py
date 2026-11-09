import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi
import jieba


class BM25Index:
    """Sparse keyword index over the chunk corpus using BM25 (Okapi).

    Built in lock-step with the vector store so that every chunk has a
    matching entry in both indexes; chunk ids are the shared key.
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = Path(persist_dir) if persist_dir else None
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._chunk_ids: List[str] = []
        self._tokenized: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._lookup: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """中文使用搜索模式分词，英文及数字保留原有的小写词项规则。"""
        tokens = []
        for part in re.findall(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]+", text.lower()):
            if part.isascii():
                tokens.append(part)
                continue
            # 查询和文档采用相同规则，长词同时保留可检索的子词。
            tokens.extend(jieba.cut_for_search(part))
        return tokens

    def add(
        self,
        chunk_ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Insert new chunks into the BM25 corpus (appends and rebuilds).

        `metadatas` is optional; if provided it is stored alongside each chunk
        so sparse retrieval can surface the same metadata as dense retrieval.
        """
        if len(chunk_ids) != len(documents):
            raise ValueError("chunk_ids and documents must be same length")
        if metadatas is not None and len(metadatas) != len(chunk_ids):
            raise ValueError("chunk_ids and metadatas must be same length")
        for i, (cid, doc) in enumerate(zip(chunk_ids, documents)):
            if cid in self._lookup:
                continue  # idempotent: skip already-indexed chunks
            self._chunk_ids.append(cid)
            toks = self.tokenize(doc)
            self._tokenized.append(toks)
            meta = metadatas[i] if metadatas is not None else {}
            self._lookup[cid] = {"tokens": toks, "content": doc, "metadata": meta}
        self._rebuild()

    def _rebuild(self) -> None:
        if any(self._tokenized):
            self._bm25 = BM25Okapi(self._tokenized)
        else:
            self._bm25 = None

    def query(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return top-k chunks ranked by BM25 score, with metadata included."""
        if self._bm25 is None or not self._tokenized:
            return []
        tokens = self.tokenize(query_text)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        results = []
        for i in ranked[:top_k]:
            cid = self._chunk_ids[i]
            results.append(
                {
                    "chunk_id": cid,
                    "score": float(scores[i]),
                    "content": self._lookup[cid]["content"],
                    "metadata": self._lookup[cid].get("metadata", {}),
                }
            )
        return results

    def count(self) -> int:
        return len(self._chunk_ids)

    def reset(self) -> None:
        """Drop all indexed chunks and remove the persisted index file."""
        self._chunk_ids = []
        self._tokenized = []
        self._lookup = {}
        self._bm25 = None
        if self.persist_dir is not None:
            path = self.persist_dir / "bm25_index.json"
            if path.exists():
                path.unlink()

    def has(self, chunk_id: str) -> bool:
        return chunk_id in self._lookup

    def save(self, file_name: str = "bm25_index.json") -> str:
        if self.persist_dir is None:
            raise RuntimeError("BM25Index initialized without persist_dir")
        path = self.persist_dir / file_name
        data = {
            "chunk_ids": self._chunk_ids,
            "tokenized": self._tokenized,
            "lookup": self._lookup,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return str(path)

    def load(self, file_name: str = "bm25_index.json") -> bool:
        if self.persist_dir is None:
            return False
        path = self.persist_dir / file_name
        if not path.exists():
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._chunk_ids = data["chunk_ids"]
        self._lookup = data["lookup"]
        # 分词结果是派生缓存；从原文恢复，避免旧索引沿用丢弃中文的规则。
        self._tokenized = []
        for chunk_id in self._chunk_ids:
            item = self._lookup[chunk_id]
            tokens = self.tokenize(item["content"])
            item["tokens"] = tokens
            self._tokenized.append(tokens)
        self._rebuild()
        return True

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
