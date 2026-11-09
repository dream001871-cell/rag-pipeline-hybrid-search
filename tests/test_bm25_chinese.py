import json

from rag_pipeline.bm25_index import BM25Index
from rag_pipeline.retrieval import SparseRetriever


def test_chinese_words_and_english_identifiers_are_preserved():
    tokens = BM25Index.tokenize("如何配置 API_KEY，处理 HTTP 429 错误？")
    assert "配置" in tokens
    assert "错误" in tokens
    assert [token for token in tokens if token.isascii()] == ["api", "key", "http", "429"]
    assert BM25Index.tokenize("WIDGET_API_KEY retry-after 429") == [
        "widget", "api", "key", "retry", "after", "429"
    ]


def test_chinese_question_retrieves_the_supporting_passage():
    index = BM25Index()
    index.add(
        ["auth", "billing", "backup"],
        ["接口认证需要配置访问令牌。", "账单每月结算一次。", "数据库每天自动备份。"],
        [{"source_file": "认证.md"}, {"source_file": "账单.md"}, {"source_file": "备份.md"}],
    )
    results = SparseRetriever(bm25_index=index).retrieve("如何配置接口认证令牌？", top_k=1)
    assert results[0].chunk_id == "auth"
    assert results[0].sparse_score > 0
    assert results[0].metadata["source_file"] == "认证.md"


def test_old_persistent_index_is_retokenized_from_original_text(tmp_path):
    path = tmp_path / "bm25_index.json"
    path.write_text(json.dumps({
        "chunk_ids": ["auth", "billing", "backup"],
        "tokenized": [[], [], []],
        "lookup": {
            "auth": {"tokens": [], "content": "接口认证需要配置访问令牌。", "metadata": {"source_file": "认证.md"}},
            "billing": {"tokens": [], "content": "账单每月结算一次。", "metadata": {}},
            "backup": {"tokens": [], "content": "数据库每天自动备份。", "metadata": {}},
        },
    }, ensure_ascii=False), encoding="utf-8")
    index = BM25Index(persist_dir=str(tmp_path))
    assert index.load()
    assert index.query("认证令牌", top_k=1)[0]["chunk_id"] == "auth"
    assert index.count() == 3
    assert index._lookup["auth"]["tokens"]
    index.save()
    restored = BM25Index(persist_dir=str(tmp_path))
    assert restored.load()
    assert restored.query("认证令牌", top_k=1)[0]["score"] > 0


def test_punctuation_only_corpus_and_queries_do_not_crash():
    index = BM25Index()
    index.add(["empty"], ["，。！？"])
    assert index.query("认证") == []
    index.add(["auth", "backup"], ["认证令牌", "每日备份"])
    assert index.query("，。！？") == []
