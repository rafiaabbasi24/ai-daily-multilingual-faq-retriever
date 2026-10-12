import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

# Configure module level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Prevent adding multiple handlers in interactive environments
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class EmbeddingService:
    """
    Service responsible for loading a transformer based sentence embedding model,
    encoding texts into dense vectors and computing similarity scores.

    The implementation is deliberately lightweight yet production‑ready:
    - Model is loaded once per process (singleton pattern via module level instance).
    - Automatic device selection (GPU if available, otherwise CPU).
    - Optional normalisation of embeddings for cosine similarity.
    - Helper methods for top‑k retrieval against a corpus.
    """

    def __init__(
        self,
        model_name_or_path: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
        cache_folder: Optional[Path] = None,
    ) -> None:
        """
        Initialise the service and load the embedding model.

        Args:
            model_name_or_path: HuggingFace model identifier or local directory.
            device: ``'cpu'`` or ``'cuda'``. If ``None`` the best available device is chosen.
            cache_folder: Optional directory for model caching.
        """
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_folder = str(cache_folder) if cache_folder else None

        logger.info(
            "Loading embedding model '%s' on device '%s'.",
            self.model_name_or_path,
            self.device,
        )
        try:
            self.model = SentenceTransformer(
                self.model_name_or_path,
                device=self.device,
                cache_folder=self.cache_folder,
            )
        except Exception as exc:
            logger.exception("Failed to load the embedding model.")
            raise RuntimeError("Could not load embedding model.") from exc

        # Verify that the model can produce embeddings
        test_embedding = self.model.encode(
            ["test"], convert_to_numpy=True, normalize_embeddings=True
        )
        if test_embedding.shape[1] == 0:
            raise ValueError("Loaded model does not produce embeddings.")
        logger.debug("Embedding model loaded successfully with dimension %d.", test_embedding.shape[1])

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into dense vectors.

        Args:
            texts: List of strings to encode.
            batch_size: Number of texts processed per forward pass.
            normalize_embeddings: If ``True``, embeddings are L2‑normalised.
            convert_to_numpy: Return a NumPy array instead of a PyTorch tensor.

        Returns:
            A 2‑D NumPy array of shape ``(len(texts), embedding_dim)``.
        """
        if not texts:
            logger.warning("Empty text list supplied to encode(); returning empty array.")
            return np.empty((0, self.model.get_sentence_embedding_dimension()))

        logger.debug(
            "Encoding %d texts (batch_size=%d, normalize=%s).",
            len(texts),
            batch_size,
            normalize_embeddings,
        )
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=convert_to_numpy,
            show_progress_bar=False,
        )
        logger.debug("Encoding completed. Shape: %s.", embeddings.shape)
        return embeddings

    @staticmethod
    def compute_cosine_similarity(
        query_embeddings: np.ndarray, corpus_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between a query embedding (or batch) and a corpus.

        Args:
            query_embeddings: Array of shape ``(n_queries, dim)``.
            corpus_embeddings: Array of shape ``(n_corpus, dim)``.

        Returns:
            A similarity matrix of shape ``(n_queries, n_corpus)`` where each entry
            is the cosine similarity between the corresponding vectors.
        """
        if query_embeddings.ndim != 2 or corpus_embeddings.ndim != 2:
            raise ValueError("Both query and corpus embeddings must be 2‑dimensional arrays.")
        logger.debug(
            "Computing cosine similarity (queries=%d, corpus=%d).",
            query_embeddings.shape[0],
            corpus_embeddings.shape[0],
        )
        similarity = cosine_similarity(query_embeddings, corpus_embeddings)
        logger.debug("Cosine similarity matrix shape: %s.", similarity.shape)
        return similarity

    def retrieve_top_k(
        self,
        query: str,
        corpus_texts: List[str],
        k: int = 5,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve the *k* most similar corpus entries for a single query.

        Args:
            query: The user query string.
            corpus_texts: List of candidate FAQ entries.
            k: Number of top results to return.
            batch_size: Batch size for encoding the corpus.
            normalize_embeddings: Whether to L2‑normalise embeddings before similarity.

        Returns:
            A list of tuples ``(corpus_text, similarity_score)`` sorted by descending similarity.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if not corpus_texts:
            logger.warning("Empty corpus supplied to retrieve_top_k(); returning empty list.")
            return []

        logger.info("Encoding query and %d corpus entries for top‑%d retrieval.", len(corpus_texts), k)

        # Encode query (single element list) and corpus
        query_emb = self.encode([query], batch_size=1, normalize_embeddings=normalize_embeddings)
        corpus_emb = self.encode(corpus_texts, batch_size=batch_size, normalize_embeddings=normalize_embeddings)

        # Compute similarity scores
        scores = self.compute_cosine_similarity(query_emb, corpus_emb).flatten()
        logger.debug("Similarity scores computed for %d corpus entries.", len(scores))

        # Get indices of top‑k scores
        top_k_idx = np.argpartition(-scores, range(min(k, len(scores))))[:k]
        # Sort the selected indices by actual score descending
        top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]

        results = [(corpus_texts[idx], float(scores[idx])) for idx in top_k_idx]
        logger.info("Top‑%d retrieval completed.", len(results))
        return results


# Singleton instance used by FastAPI dependency injection or other modules
_embedding_service_instance: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """
    Retrieve a global ``EmbeddingService`` instance.

    This function ensures that the heavy model is loaded only once per process.
    It can be used directly or as a FastAPI dependency.

    Returns:
        An ``EmbeddingService`` object.
    """
    global _embedding_service_instance
    if _embedding_service_instance is None:
        logger.info("Creating a new global EmbeddingService instance.")
        _embedding_service_instance = EmbeddingService()
    else:
        logger.debug("Reusing existing global EmbeddingService instance.")
    return _embedding_service_instance