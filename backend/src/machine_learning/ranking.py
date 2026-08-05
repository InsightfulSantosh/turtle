from __future__ import annotations

from typing import Any, Mapping, Sequence

from domain.contracts import Candidate, RankedCandidate


class CandidateReranker:
    def __init__(self, model_path: str | None = None):
        # Kept for constructor compatibility; learned/attribute reranking is
        # deliberately disabled by the visual-only policy.
        del model_path
        self.model_name = "visual-only"

    def rank(
        self,
        product: Mapping[str, Any],
        candidates: Sequence[Candidate],
        top_k: int = 10,
    ) -> list[RankedCandidate]:
        item_type = str(product.get("itemType") or "")
        eligible = [
            candidate
            for candidate in candidates
            if str(candidate.item.get("itemType") or "") == item_type
        ]
        ranked = [
            RankedCandidate(
                candidate,
                float(candidate.vector_similarity),
                0.0,
                {"vector_similarity": float(candidate.vector_similarity)},
            )
            for candidate in eligible
        ]
        ranked.sort(key=lambda value: value.score, reverse=True)
        return ranked[: min(max(top_k, 1), 50)]
