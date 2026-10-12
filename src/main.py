import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------------------------------- #
# Data handling utilities
# --------------------------------------------------------------------------- #
def load_faqs(file_path: Path) -> List[Dict[str, str]]:
    """
    Load FAQs from a JSON or CSV file.

    Supported formats:
        * JSON – a list of objects with ``question`` and ``answer`` keys.
        * CSV – with header ``question,answer``.

    Parameters
    ----------
    file_path: Path
        Path to the FAQ file.

    Returns
    -------
    List[Dict[str, str]]
        List of dictionaries, each containing ``question`` and ``answer``.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"FAQ file not found: {file_path}")

    if file_path.suffix.lower() == ".json":
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON FAQ file must contain a list of objects.")
        for entry in data:
            if "question" not in entry or "answer" not in entry:
                raise ValueError("Each FAQ entry must contain 'question' and 'answer' keys.")
        return data

    elif file_path.suffix.lower() == ".csv":
        import csv

        faqs = []
        with file_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "question" not in reader.fieldnames or "answer" not in reader.fieldnames:
                raise ValueError("CSV must contain 'question' and 'answer' columns.")
            for row in reader:
                faqs.append({"question": row["question"], "answer": row["answer"]})
        return faqs

    else:
        raise ValueError("Unsupported FAQ file format. Use .json or .csv")


def embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """
    Compute embeddings for a list of texts using a SentenceTransformer model.

    Parameters
    ----------
    texts: List[str]
        Texts to embed.
    model: SentenceTransformer
        Pre‑loaded sentence‑transformer model.

    Returns
    -------
    np.ndarray
        2‑D array of shape (len(texts), embedding_dim).
    """
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()))
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    # Normalise to unit length for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def build_index(embeddings: np.ndarray) -> NearestNeighbors:
    """
    Build a cosine‑similarity nearest‑neighbors index.

    Parameters
    ----------
    embeddings: np.ndarray
        Normalised embeddings.

    Returns
    -------
    NearestNeighbors
        Fitted sklearn NearestNeighbors model.
    """
    nn = NearestNeighbors(metric="cosine", algorithm="auto")
    nn.fit(embeddings)
    return nn


def save_artifacts(
    index: NearestNeighbors,
    embeddings: np.ndarray,
    faqs: List[Dict[str, str]],
    destination: Path,
) -> None:
    """
    Persist the index, embeddings and raw FAQs to disk.

    Parameters
    ----------
    index: NearestNeighbors
        Fitted nearest‑neighbors model.
    embeddings: np.ndarray
        Embedding matrix.
    faqs: List[Dict[str, str]]
        Original FAQ data.
    destination: Path
        Directory where artefacts will be stored.
    """
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, destination / "nn_index.joblib")
    np.save(destination / "embeddings.npy", embeddings)
    with (destination / "faqs.json").open("w", encoding="utf-8") as f:
        json.dump(faqs, f, ensure_ascii=False, indent=2)


def load_artifacts(source: Path) -> Tuple[NearestNeighbors, np.ndarray, List[Dict[str, str]]]:
    """
    Load previously saved index, embeddings and FAQs.

    Parameters
    ----------
    source: Path
        Directory containing the persisted artefacts.

    Returns
    -------
    Tuple[NearestNeighbors, np.ndarray, List[Dict[str, str]]]
        The loaded nearest‑neighbors model, embedding matrix and FAQ list.
    """
    if not source.is_dir():
        raise FileNotFoundError(f"Index directory not found: {source}")

    index = joblib.load(source / "nn_index.joblib")
    embeddings = np.load(source / "embeddings.npy")
    with (source / "faqs.json").open("r", encoding="utf-8") as f:
        faqs = json.load(f)
    return index, embeddings, faqs


# --------------------------------------------------------------------------- #
# Query handling
# --------------------------------------------------------------------------- #
def query_faq(
    query: str,
    model: SentenceTransformer,
    nn: NearestNeighbors,
    faqs: List[Dict[str, str]],
    top_k: int = 5,
) -> List[Tuple[float, Dict[str, str]]]:
    """
    Retrieve the most similar FAQs for a given query.

    Parameters
    ----------
    query: str
        User question.
    model: SentenceTransformer
        Embedding model.
    nn: NearestNeighbors
        Pre‑built index over FAQ embeddings.
    faqs: List[Dict[str, str]]
        Original FAQ data.
    top_k: int, default 5
        Number of results to return.

    Returns
    -------
    List[Tuple[float, Dict[str, str]]]
        List of (distance, faq) tuples sorted by increasing distance.
    """
    query_emb = embed_texts([query], model)  # shape (1, dim)
    distances, indices = nn.kneighbors(query_emb, n_neighbors=top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append((float(dist), faqs[int(idx)]))
    return results


# --------------------------------------------------------------------------- #
# FastAPI service
# --------------------------------------------------------------------------- #
app = FastAPI(title="Multilingual FAQ Retrieval")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class Answer(BaseModel):
    question: str
    answer: str
    score: float


class QueryResponse(BaseModel):
    results: List[Answer]


# Global objects initialised at startup
_GLOBAL_MODEL: SentenceTransformer | None = None
_GLOBAL_NN: NearestNeighbors | None = None
_GLOBAL_FAQS: List[Dict[str, str]] = []


@app.on_event("startup")
def _load_resources() -> None:
    """
    Load model, index and FAQs when the FastAPI app starts.
    Environment variables are used to locate resources:

        * ``FAQ_MODEL`` – model name or path (default: multilingual MiniLM)
        * ``FAQ_INDEX_DIR`` – directory containing saved artefacts
    """
    global _GLOBAL_MODEL, _GLOBAL_NN, _GLOBAL_FAQS

    model_name = os.getenv("FAQ_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    index_dir = os.getenv("FAQ_INDEX_DIR")
    if not index_dir:
        raise RuntimeError("Environment variable FAQ_INDEX_DIR must be set to the index directory.")

    _GLOBAL_MODEL = SentenceTransformer(model_name)
    _GLOBAL_NN, _, _GLOBAL_FAQS = load_artifacts(Path(index_dir))


@app.post("/query", response_model=QueryResponse)
def api_query(payload: QueryRequest) -> QueryResponse:
    """
    FastAPI endpoint to query the FAQ store.

    Parameters
    ----------
    payload: QueryRequest
        JSON payload containing ``question`` and optional ``top_k``.

    Returns
    -------
    QueryResponse
        List of matching FAQs with similarity scores.
    """
    if _GLOBAL_MODEL is None or _GLOBAL_NN is None:
        raise HTTPException(status_code=500, detail="Model or index not loaded.")

    results = query_faq(
        query=payload.question,
        model=_GLOBAL_MODEL,
        nn=_GLOBAL_NN,
        faqs=_GLOBAL_FAQS,
        top_k=payload.top_k,
    )
    answers = [
        Answer(question=res[1]["question"], answer=res[1]["answer"], score=1 - res[0])
        for res in results
    ]
    return QueryResponse(results=answers)


# --------------------------------------------------------------------------- #
# Command‑line interface
# --------------------------------------------------------------------------- #
def _cmd_build(args: argparse.Namespace) -> None:
    """
    Build the FAQ index from a source file and persist it.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed command‑line arguments.
    """
    faqs = load_faqs(Path(args.faq_path))
    model = SentenceTransformer(args.model_name)
    questions = [faq["question"] for faq in faqs]
    embeddings = embed_texts(questions, model)
    nn = build_index(embeddings)
    save_artifacts(nn, embeddings, faqs, Path(args.output_dir))
    print(f"Index built and saved to '{args.output_dir}'.")


def _cmd_query(args: argparse.Namespace) -> None:
    """
    Perform a single query against a pre‑built index.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed command‑line arguments.
    """
    model = SentenceTransformer(args.model_name)
    nn, _, faqs = load_artifacts(Path(args.index_dir))
    results = query_faq(args.question, model, nn, faqs, top_k=args.top_k)
    for score, faq in results:
        similarity = 1 - score  # convert cosine distance to similarity
        print(f"[{similarity:.4f}] Q: {faq['question']}\n    A: {faq['answer']}\n")


def _cmd_serve(args: argparse.Namespace) -> None:
    """
    Launch the FastAPI server.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed command‑line arguments.
    """
    # Export required environment variables for the FastAPI startup hook
    os.environ["FAQ_MODEL"] = args.model_name
    os.environ["FAQ_INDEX_DIR"] = str(Path(args.index_dir).resolve())
    uvicorn.run("src.main:app", host=args.host, port=args.port, reload=args.reload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multilingual FAQ Retrieval with Transformer Embeddings"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build-index sub‑command
    build_parser = subparsers.add_parser(
        "build-index", help="Create an index from a FAQ file."
    )
    build_parser.add_argument(
        "faq_path", type=str, help="Path to the FAQ source file (JSON or CSV)."
    )
    build_parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model identifier.",
    )
    build_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the index and embeddings will be stored.",
    )
    build_parser.set_defaults(func=_cmd_build)

    # query sub‑command
    query_parser = subparsers.add_parser(
        "query", help="Run a single query against a pre‑built index."
    )
    query_parser.add_argument(
        "question", type=str, help="User question to retrieve similar FAQs for."
    )
    query_parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model identifier.",
    )
    query_parser.add_argument(
        "--index-dir",
        type=str,
        required=True,
        help="Directory containing the previously built index.",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of FAQ results to return.",
    )
    query_parser.set_defaults(func=_cmd_query)

    # serve sub‑command
    serve_parser = subparsers.add_parser(
        "serve", help="Start a FastAPI server exposing the FAQ retrieval API."
    )
    serve_parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model identifier.",
    )
    serve_parser.add_argument(
        "--index-dir",
        type=str,
        required=True,
        help="Directory containing the previously built index.",
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address for the FastAPI server.",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the FastAPI server.",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto‑reload (useful during development).",
    )
    serve_parser.set_defaults(func=_cmd_serve)

    return parser


def main(argv: List[str] | None = None) -> None:
    """
    Entry point for the command‑line interface.

    Parameters
    ----------
    argv: List[str] | None
        Optional argument list; defaults to ``sys.argv[1:]``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()