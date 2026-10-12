"""Integration tests for the FastAPI FAQ retrieval service.

The tests cover the main public endpoints:
- GET /health
- POST /faqs (create a new FAQ)
- GET /faqs/{faq_id} (retrieve a stored FAQ)
- POST /search (semantic search over stored FAQs)

The test suite uses FastAPI's TestClient and runs against the in‑memory
application instance defined in ``app.main``.  No external services are
required; the embedding model is the lightweight ``sentence‑transformers``
model bundled with the project.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Import the FastAPI application. Adjust the import path if the project
# structure differs.
from app.main import app

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a TestClient instance for the duration of the test module."""
    with TestClient(app) as c:
        yield c

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def create_faq(client: TestClient, question: str, answer: str, language: str = "en") -> int:
    """Create an FAQ via the API and return its identifier.

    The function asserts that the creation request succeeded and extracts the
    ``faq_id`` from the JSON response.
    """
    payload = {"question": question, "answer": answer, "language": language}
    response = client.post("/faqs", json=payload)
    assert response.status_code == 201, f"FAQ creation failed: {response.text}"
    data = response.json()
    assert "faq_id" in data, "Response missing 'faq_id'"
    return int(data["faq_id"])

# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_health_endpoint(client: TestClient) -> None:
    """The health endpoint must return a simple JSON payload indicating OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_create_and_retrieve_faq(client: TestClient) -> None:
    """Create an FAQ and then retrieve it, checking that the stored data matches."""
    question = "What is the refund policy?"
    answer = "You can request a refund within 30 days of purchase."
    language = "en"

    faq_id = create_faq(client, question, answer, language)

    # Retrieve the newly created FAQ
    response = client.get(f"/faqs/{faq_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["faq_id"] == faq_id
    assert data["question"] == question
    assert data["answer"] == answer
    assert data["language"] == language

def test_search_returns_relevant_faq(client: TestClient) -> None:
    """Semantic search should return FAQs that are relevant to the query."""
    # Ensure there is at least one known FAQ in the store.
    q1 = "How do I reset my password?"
    a1 = "Click the 'Forgot password' link on the login page and follow the instructions."
    faq_id_1 = create_faq(client, q1, a1, language="en")

    # Perform a search with a semantically similar query.
    query = "I forgot my password, how can I change it?"
    response = client.post("/search", json={"query": query, "language": "en"})
    assert response.status_code == 200, f"Search failed: {response.text}"
    results = response.json()
    # The API is expected to return a list of result objects.
    assert isinstance(results, list), "Search response is not a list"
    # At least one result should be returned.
    assert len(results) > 0, "Search returned no results"

    # The most relevant result should be the FAQ we just inserted.
    top_result = results[0]
    assert "faq_id" in top_result
    assert top_result["faq_id"] == faq_id_1
    # The score field is optional but, if present, should be a float.
    if "score" in top_result:
        assert isinstance(top_result["score"], float)

def test_create_faq_validation_error(client: TestClient) -> None:
    """Submitting an incomplete payload must result in a 422 validation error."""
    # Missing the ``answer`` field.
    payload = {"question": "What is the support email?", "language": "en"}
    response = client.post("/faqs", json=payload)
    assert response.status_code == 422
    error_detail = response.json()
    # FastAPI returns a ``detail`` key with a list of validation errors.
    assert "detail" in error_detail
    assert any(err["loc"][-1] == "answer" for err in error_detail["detail"])

def test_search_no_results(client: TestClient) -> None:
    """A query that does not match any stored FAQ should return an empty list."""
    response = client.post(
        "/search",
        json={"query": "Quantum entanglement in bananas", "language": "en"},
    )
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert len(results) == 0

# --------------------------------------------------------------------------- #
# Clean‑up (optional)
# --------------------------------------------------------------------------- #

def test_cleanup_faqs(client: TestClient) -> None:
    """If the application provides a bulk delete endpoint, clear the store.

    This test is optional and will be a no‑op if the endpoint does not exist.
    It ensures that subsequent test runs start from a clean state.
    """
    # Attempt to delete all FAQs; ignore 404/405 errors gracefully.
    response = client.delete("/faqs")
    if response.status_code not in (200, 204, 404, 405):
        pytest.fail(f"Unexpected status code from cleanup endpoint: {response.status_code}")