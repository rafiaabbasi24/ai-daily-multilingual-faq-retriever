"""tests/test_main.py
Unit tests for the multilingual FAQ retrieval logic.

The tests cover:
- Proper construction of the FAQRetriever.
- Correct encoding of FAQ questions.
- Accurate nearest‑neighbor search across languages.
- Deterministic behavior and reasonable similarity scores.

The implementation assumes the project provides a ``FAQRetriever`` class in
``faq_retriever.py`` with the following public API:

* ``FAQRetriever(faqs: List[Dict[str, str]], model_name: str = ...)``
* ``build_index()`` – computes embeddings and builds a nearest‑neighbor index.
* ``encode_questions() -> np.ndarray`` – returns the matrix of question embeddings.
* ``search(query: str, top_k: int = 1) -> List[Dict[str, Any]]`` – returns a list
  of the most similar FAQ entries, each containing ``question``, ``answer`` and
  ``score`` keys.

If the actual API differs, adjust the imports and method calls accordingly.
"""

from __future__ import annotations

import numpy as np
import pytest

# The project should expose a ``FAQRetriever`` class. Adjust the import path if
# the class lives in a different module (e.g. ``src.faq_retriever``).
from faq_retriever import FAQRetriever


@pytest.fixture(scope="module")
def sample_faqs() -> list[dict[str, str]]:
    """A small multilingual FAQ dataset used for all tests."""
    return [
        {
            "question": "How can I reset my password?",
            "answer": "Click on 'Forgot password' and follow the instructions.",
        },
        {
            "question": "¿Cómo puedo restablecer mi contraseña?",
            "answer": "Haz clic en 'Olvidé mi contraseña' y sigue las instrucciones.",
        },
        {
            "question": "Comment réinitialiser mon mot de passe ?",
            "answer": "Cliquez sur 'Mot de passe oublié' et suivez les instructions.",
        },
    ]


@pytest.fixture(scope="module")
def retriever(sample_faqs: list[dict[str, str]]) -> FAQRetriever:
    """
    Initialise a ``FAQRetriever`` with a lightweight multilingual model.
    The fixture builds the index once for the entire test module.
    """
    # The MiniLM multilingual model is small enough for CI while still
    # providing cross‑lingual capabilities.
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    r = FAQRetriever(faqs=sample_faqs, model_name=model_name)
    r.build_index()
    return r


def test_embeddings_shape(retriever: FAQRetriever, sample_faqs: list[dict[str, str]]) -> None:
    """
    Verify that the embedding matrix has the expected shape:
    (number_of_faqs, embedding_dim).
    """
    embeddings = retriever.encode_questions()
    assert isinstance(embeddings, np.ndarray), "Embeddings should be a NumPy array"
    assert embeddings.shape[0] == len(sample_faqs), "One embedding per FAQ entry"
    # The embedding dimension is model‑dependent; we only assert it is > 0.
    assert embeddings.shape[1] > 0, "Embedding dimension must be positive"


@pytest.mark.parametrize(
    "query,expected_answer",
    [
        (
            "I forgot my password, how do I change it?",
            "Click on 'Forgot password' and follow the instructions.",
        ),
        (
            "Olvidé mi contraseña, ¿qué debo hacer?",
            "Haz clic en 'Olvidé mi contraseña' y sigue las instrucciones.",
        ),
        (
            "Comment je peux changer mon mot de passe ?",
            "Cliquez sur 'Mot de passe oublié' et suivez les instructions.",
        ),
    ],
)
def test_cross_lingual_search(
    retriever: FAQRetriever, query: str, expected_answer: str
) -> None:
    """
    Ensure that a query in any of the supported languages retrieves the correct
    FAQ answer. The test also checks that the similarity score is reasonably
    high (>= 0.5) for the top‑1 result.
    """
    results = retriever.search(query, top_k=1)
    assert isinstance(results, list) and len(results) == 1, "search should return a list with one result"
    top_result = results[0]

    # The result dictionary must contain the expected keys.
    for key in ("question", "answer", "score"):
        assert key in top_result, f"Result missing expected key: {key}"

    # Verify that the retrieved answer matches the expected language‑specific answer.
    assert top_result["answer"] == expected_answer

    # The similarity score is model‑dependent but should be well above random.
    assert isinstance(top_result["score"], float)
    assert top_result["score"] >= 0.5, f"Score too low ({top_result['score']:.3f}) for query: {query}"


def test_search_returns_multiple_results(retriever: FAQRetriever) -> None:
    """
    When ``top_k`` > 1 the retriever should return that many distinct entries,
    ordered by decreasing similarity.
    """
    query = "Password reset instructions"
    top_k = 3
    results = retriever.search(query, top_k=top_k)

    assert isinstance(results, list)
    assert len(results) == top_k, f"Expected {top_k} results, got {len(results)}"

    # Ensure ordering by score (descending)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Results are not sorted by descending score"

    # All returned entries must belong to the original FAQ set.
    original_questions = {faq["question"] for faq in retriever.faqs}
    for r in results:
        assert r["question"] in original_questions, "Returned question not part of the original FAQ set"


def test_invalid_query_raises_error(retriever: FAQRetriever) -> None:
    """
    Passing a non‑string query should raise a ``TypeError``.
    """
    with pytest.raises(TypeError):
        retriever.search(12345)  # type: ignore[arg-type]


def test_empty_faq_list_raises_error() -> None:
    """
    Constructing a ``FAQRetriever`` with an empty FAQ list should raise a
    ``ValueError`` because there is nothing to index.
    """
    with pytest.raises(ValueError):
        FAQRetriever(faqs=[], model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")  # type: ignore[arg-type]