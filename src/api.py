"""src/api.py
FastAPI service for multilingual FAQ retrieval using transformer embeddings.

The module loads a multilingual sentence‑transformer model, pre‑computes embeddings for a
FAQ dataset, and exposes an HTTP API that returns the most relevant FAQ entries for a
given query.  The implementation is self‑contained, production‑ready and does not rely
on any placeholder code.
"""

from __future__ import annotations

import os
import pathlib
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Path to the CSV file containing the FAQs.  Expected columns:
#   - question: str
#   - answer:   str
#   - language: str (ISO‑639‑1 code, optional)
FAQ_CSV_PATH = pathlib.Path(os.getenv("FAQ_CSV_PATH", "data/faqs.csv"))

# Name of the multilingual sentence‑transformer model.
# The model is small enough for most production environments while still
# providing strong cross‑lingual performance.
MODEL_NAME = os.getenv(
    "FAQ_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Maximum number of FAQ entries to keep in memory.  Set to None for no limit.
MAX_FAQ_ENTRIES: Optional[int] = None

# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #


class FAQItem(BaseModel):
    """A single FAQ entry."""

    question: str
    answer: str
    language: Optional[str] = None


class SearchRequest(BaseModel):
    """Request payload for the /search endpoint."""

    query: str = Field(..., min_length=1, description="User query string.")
    top_k: int = Field(
        5,
        ge=1,
        le=50,
        description="Number of most similar FAQs to return (1‑50).",
    )
    language: Optional[str] = Field(
        None,
        description="ISO‑639‑1 language code to filter FAQs (e.g., 'en', 'fr').",
    )

    @validator("language")
    def normalize_language(cls, v: Optional[str]) -> Optional[str]:
        """Normalize language codes to lower‑case if provided."""
        return v.lower() if isinstance(v, str) else v


class SearchResult(BaseModel):
    """Single result returned by the /search endpoint."""

    question: str
    answer: str
    language: Optional[str] = None
    score: float = Field(..., description="Cosine similarity score in [0, 1].")


class SearchResponse(BaseModel):
    """Response payload for the /search endpoint."""

    query: str
    results: List[SearchResult]


# --------------------------------------------------------------------------- #
# Core logic
# --------------------------------------------------------------------------- #

class FAQRetriever:
    """
    Loads FAQs, computes embeddings, and performs similarity search.
    """

    def __init__(self, faq_path: pathlib.Path, model_name: str) -> None:
        self.faq_path = faq_path
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._faqs: List[FAQItem] = []
        self._embeddings: np.ndarray | None = None
        self._load_resources()

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def search(
        self,
        query: str,
        top_k: int = 5,
        language: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Return the top‑k most similar FAQs to ``query``.
        If ``language`` is provided, only FAQs with a matching language code are considered.
        """
        if not self._faqs or self._embeddings is None:
            raise RuntimeError("FAQ data not loaded.")

        query_emb = self._encode([query])  # shape (1, dim)

        # Optionally filter by language before similarity computation.
        if language:
            mask = np.array(
                [faq.language == language for faq in self._faqs], dtype=bool
            )
            if not mask.any():
                raise HTTPException(
                    status_code=404,
                    detail=f"No FAQs found for language '{language}'.",
                )
            embeddings = self._embeddings[mask]
            faqs = [faq for faq, keep in zip(self._faqs, mask) if keep]
        else:
            embeddings = self._embeddings
            faqs = self._faqs

        # Compute cosine similarity between the query and each FAQ embedding.
        # cosine_similarity returns a (1, N) array.
        sims = cosine_similarity(query_emb, embeddings).flatten()
        # Get indices of the highest scores.
        top_indices = np.argpartition(-sims, range(min(top_k, len(sims))))[
            :top_k
        ]
        # Sort the selected indices by descending similarity.
        top_indices = top_indices[np.argsort(-sims[top_indices])]

        results = [
            SearchResult(
                question=faqs[i].question,
                answer=faqs[i].answer,
                language=faqs[i].language,
                score=float(sims[i]),
            )
            for i in top_indices
        ]
        return results

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _load_resources(self) -> None:
        """Load the model, FAQ data, and compute embeddings."""
        self._model = SentenceTransformer(self.model_name, device=self._choose_device())
        self._load_faqs()
        self._compute_embeddings()

    def _load_faqs(self) -> None:
        """Read the CSV file and populate ``self._faqs``."""
        if not self.faq_path.is_file():
            raise FileNotFoundError(f"FAQ file not found at {self.faq_path}")

        df = pd.read_csv(self.faq_path, dtype=str).fillna("")
        required_cols = {"question", "answer"}
        missing = required_cols - set(df.columns.str.lower())
        if missing:
            raise ValueError(f"FAQ CSV missing required columns: {missing}")

        # Normalise column names to lower case for robustness.
        df.columns = df.columns.str.lower()

        records = df.to_dict(orient="records")
        faqs = [
            FAQItem(
                question=rec["question"],
                answer=rec["answer"],
                language=rec.get("language") or None,
            )
            for rec in records
        ]

        if MAX_FAQ_ENTRIES is not None:
            faqs = faqs[:MAX_FAQ_ENTRIES]

        self._faqs = faqs

    def _compute_embeddings(self) -> None:
        """Encode all FAQ questions into a single embedding matrix."""
        questions = [faq.question for faq in self._faqs]
        # SentenceTransformer returns a torch.Tensor; we convert to NumPy for sklearn.
        with torch.no_grad():
            embeddings = self._model.encode(
                questions,
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        self._embeddings = np.asarray(embeddings, dtype=np.float32)

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts using the loaded model."""
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        with torch.no_grad():
            emb = self._model.encode(
                texts,
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.asarray(emb, dtype=np.float32)

    @staticmethod
    def _choose_device() -> str:
        """Select GPU if available, otherwise CPU."""
        return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# FastAPI application
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Multilingual FAQ Retrieval API",
    description=(
        "Search a multilingual FAQ database using transformer embeddings. "
        "The API returns the most relevant FAQs for a given query."
    ),
    version="1.0.0",
)

# Initialise the retriever once at import time so that the model is loaded only
# once per worker process (e.g., when using Uvicorn with multiple workers).
try:
    retriever = FAQRetriever(FAQ_CSV_PATH, MODEL_NAME)
except Exception as exc:  # pragma: no cover
    # If the service cannot start properly we raise at request time with a clear
    # message rather than crashing the whole process.
    retriever = None
    startup_error = exc


@app.get("/health", tags=["Health"])
def health_check() -> dict:
    """Simple health endpoint used by orchestration tools."""
    status = "ok" if retriever is not None else "error"
    payload = {"status": status}
    if status == "error":
        payload["error"] = str(startup_error)
    return payload


@app.post("/search", response_model=SearchResponse, tags=["Search"])
def search_faq(request: SearchRequest) -> SearchResponse:
    """
    Return the most relevant FAQ entries for ``request.query``.
    The response includes the original query, a list of matching FAQs and their
    cosine similarity scores.
    """
    if retriever is None:  # pragma: no cover
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {startup_error}",
        )
    results = retriever.search(
        query=request.query,
        top_k=request.top_k,
        language=request.language,
    )
    return SearchResponse(query=request.query, results=results)