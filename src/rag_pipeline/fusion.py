from typing import List, Dict, Tuple, Optional

from .retrieval import RetrievalResult, DenseRetriever, SparseRetriever
from .reranker import Reranker, RerankerFactory

DEFAULT_TOP_K = 20


class ReciprocalRankFusion:
    """Combines multiple ranked result lists into one using Reciprocal Rank
    Fusion (RRF).

    RRF score for a document across rank lists:
        RRF(d) = sum over lists L of  w_L * 1 / (k + rank_L(d))

    where `k` is a smoothing constant (standard value 60) and `w_L` is a
    per-list weight (e.g. 0.7 dense / 0.3 sparse). Weighting is configurable so
    it can be tuned per use case. Because RRF is rank-based, it is robust to
    the incomparable absolute scores of cosine similarity vs BM25.
    """

    def __init__(self, k: int = 60):
        if k <= 0:
            raise ValueError("RRF constant k must be positive")
        self.k = k

    def fuse(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        weight_dense: float = 0.7,
        weight_sparse: float = 0.3,
    ) -> List[RetrievalResult]:
        """Merge dense and sparse results into one RRF-ranked list.

        A document's rank within a list is its 1-based position. Documents
        contributed by a single list still participate; documents from both
        lists get the benefit of both rank contributions.
        """
        # Accumulate per-chunk RRF score + the metadata/content of the first
        # occurrence (they should be identical across lists for the same id).
        rrf_scores: Dict[str, float] = {}
        content_by_id: Dict[str, str] = {}
        meta_by_id: Dict[str, dict] = {}
        dense_by_id: Dict[str, float] = {}
        sparse_by_id: Dict[str, float] = {}

        for rank, res in enumerate(dense_results, start=1):
            cid = res.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + weight_dense * (
                1.0 / (self.k + rank)
            )
            content_by_id.setdefault(cid, res.content)
            meta_by_id.setdefault(cid, res.metadata)
            dense_by_id.setdefault(cid, res.dense_score)

        for rank, res in enumerate(sparse_results, start=1):
            cid = res.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + weight_sparse * (
                1.0 / (self.k + rank)
            )
            content_by_id.setdefault(cid, res.content)
            meta_by_id.setdefault(cid, res.metadata)
            sparse_by_id.setdefault(cid, res.sparse_score)

        # Sort by RRF score descending.
        ranked = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)

        fused = []
        for cid, score in ranked:
            result = RetrievalResult(
                chunk_id=cid,
                content=content_by_id[cid],
                metadata=meta_by_id[cid],
                dense_score=dense_by_id.get(cid),
                sparse_score=sparse_by_id.get(cid),
            )
            result.rrf_score = round(score, 6)
            fused.append(result)
        return fused


class HybridRetriever:
    """End-to-end hybrid retrieval: dense + sparse -> Reciprocal Rank Fusion.

    Runs the query through both the dense (vector) and sparse (BM25) retriever
    in parallel, fuses their ranked lists with configurable RRF weights, and
    returns a single ranked list. Both retrievers can also be retrieved
    separately for comparison/diagnostics.
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        fusion: Optional[ReciprocalRankFusion] = None,
        reranker: Optional[Reranker] = None,
        dense_k: int = 60,
        sparse_k: int = 60,
        candidate_k: int = 60,
        rerank_enabled: bool = True,
        weight_dense: float = 0.7,
        weight_sparse: float = 0.3,
    ):
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.sparse_retriever = sparse_retriever or SparseRetriever()
        self.fusion = fusion or ReciprocalRankFusion()
        self.reranker = reranker or RerankerFactory().get_reranker()
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.candidate_k = candidate_k
        self.rerank_enabled = rerank_enabled
        self.weight_dense = weight_dense
        self.weight_sparse = weight_sparse

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        dense_k: Optional[int] = None,
        sparse_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
        weight_dense: Optional[float] = None,
        weight_sparse: Optional[float] = None,
        rerank: Optional[bool] = None,
    ) -> List[RetrievalResult]:
        """B 方案：每路默认召回60段，融合后重排60段，默认返回前20段。

        显式参数继续用于诊断与外部调用，rerank=False 时直接返回融合排名。
        """
        d_k = dense_k or self.dense_k
        s_k = sparse_k or self.sparse_k
        wd = weight_dense if weight_dense is not None else self.weight_dense
        ws = weight_sparse if weight_sparse is not None else self.weight_sparse
        c_k = candidate_k or self.candidate_k
        do_rerank = self.rerank_enabled if rerank is None else rerank

        dense_results = self.dense_retriever.retrieve(query, top_k=d_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=s_k)

        fused = self.fusion.fuse(
            dense_results, sparse_results, weight_dense=wd, weight_sparse=ws
        )

        # Send top candidates through the reranker, then keep top_k.
        candidates = fused[:c_k]
        if do_rerank and candidates:
            reranked = self.reranker.score(query, candidates)
            return reranked[:top_k]
        return fused[:top_k]

    def retrieve_dense(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        return self.dense_retriever.retrieve(query, top_k=top_k or self.dense_k)

    def retrieve_sparse(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        return self.sparse_retriever.retrieve(query, top_k=top_k or self.sparse_k)
