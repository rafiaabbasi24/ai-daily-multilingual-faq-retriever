# Multilingual FAQ Retrieval with Transformer Embeddings

## Overview
This project implements a multilingual FAQ retrieval system that leverages transformer‑based sentence embeddings to match user queries with the most relevant FAQ entries across multiple languages. It combines **PyTorch**, **Hugging Face Transformers**, and **sentence‑transformers** for embedding generation, **FastAPI** for serving the model as a REST API, and **scikit‑learn** for efficient similarity search.

Key capabilities:
- **Multilingual support** (English, Spanish, French, German, Chinese, etc.) using a single multilingual model.
- **Real‑time inference** via a lightweight FastAPI service.
- **Scalable indexing** with FAISS or scikit‑learn’s `NearestNeighbors`.
- **Easy extensibility** for new languages, FAQ datasets, or custom transformer models.

## Architecture
```
└── src
    ├── data/
    │   └── faq_dataset.csv          # FAQ entries (question, answer, language)
    ├── embeddings/
    │   └── embedder.py              # SentenceTransformer wrapper
    ├── indexing/
    │   └── indexer.py               # Build / query FAISS / NearestNeighbors index
    ├── api/
    │   └── main.py                  # FastAPI application
    └── utils/
        └── config.py                # Central configuration (model name, device, etc.)
```

- **Embedding generation**: `EmbeddingModel` loads a multilingual sentence‑transformer (e.g., `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) and produces dense vectors for FAQ questions.
- **Indexing**: The vectors are stored in a scikit‑learn `NearestNeighbors` index (fallback) or FAISS index for large corpora.
- **API**: `POST /search` accepts a JSON payload with a query string and optional language code, returning the top‑k most similar FAQs.

## Tech Stack
- **Python ≥3.9**
- **PyTorch**
- **Transformers** (🤗)
- **sentence‑transformers**
- **FastAPI**
- **scikit‑learn**
- **Uvicorn** (ASGI server)
- **FAISS** (optional, for large‑scale indexing)

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/multilingual-faq-retrieval.git
cd multilingual-faq-retrieval
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note**: If you plan to use FAISS for large datasets, install the GPU‑compatible version:
> ```bash
> pip install faiss-gpu   # or faiss-cpu for CPU‑only
> ```

### 4. (Optional) Install development tools
```bash
pip install -r dev-requirements.txt
pre-commit install
```

## Quick Start

### 1. Prepare the FAQ dataset
Place a CSV file named `faq_dataset.csv` in `src/data/` with the following columns:
- `question` – the FAQ question text.
- `answer`   – the corresponding answer.
- `language` – ISO‑639‑1 language code (e.g., `en`, `es`, `fr`).

### 2. Build the embedding index
```bash
python -m src.indexing.build_index \
    --data_path src/data/faq_dataset.csv \
    --model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    --index_path models/faq_index.pkl
```
This command:
1. Loads the multilingual transformer.
2. Computes embeddings for all questions.
3. Persists a scikit‑learn `NearestNeighbors` index to `models/faq_index.pkl`.

### 3. Run the API server
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The service will be reachable at `http://localhost:8000`.

### 4. Test a query
```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "How can I reset my password?", "language": "en", "top_k": 3}'
```

Expected JSON response:
```json
{
  "results": [
    {
      "question": "How do I change my password?",
      "answer": "Go to Settings → Security → Change Password.",
      "score": 0.92
    },
    {
      "question": "I forgot my password, what should I do?",
      "answer": "Use the 'Forgot password' link on the login page.",
      "score": 0.88
    },
    ...
  ]
}
```

## Detailed Usage

### Configuration
All configurable parameters live in `src/utils/config.py`. Common settings include:
- `MODEL_NAME`: default transformer model.
- `DEVICE`: `"cuda"` if a GPU is available, otherwise `"cpu"`.
- `INDEX_TYPE`: `"sklearn"` or `"faiss"`.

You can override any setting via environment variables, e.g.:
```bash
export MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
export INDEX_TYPE=faiss
```

### API Reference

| Endpoint | Method | Request Body | Description |
|----------|--------|--------------|-------------|
| `/search` | `POST` | `{ "query": str, "language": str (optional), "top_k": int (default=5) }` | Returns the top‑k most similar FAQs. |
| `/health` | `GET` | – | Simple health check (returns `{"status":"ok"}`). |
| `/reload` | `POST` | – | Reloads the embedding index without restarting the server (admin only). |

### Adding New Languages
1. Ensure the chosen multilingual model supports the target language (check the model card on Hugging Face).
2. Append new FAQ rows with the appropriate `language` code.
3. Re‑run the `build_index` script to refresh embeddings.

### Switching to FAISS
If your FAQ corpus exceeds ~10k entries, FAISS provides faster nearest‑neighbor search:
```bash
python -m src.indexing.build_index \
    --index_type faiss \
    --faiss_metric inner_product \
    --faiss_nlist 100
```
Corresponding changes in `src/api/main.py` are automatically handled based on `INDEX_TYPE`.

## Testing
Run the test suite with:
```bash
pytest -v
```
The tests cover:
- Embedding generation consistency.
- Index building and query correctness.
- FastAPI endpoint responses and validation.

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/awesome-feature`).
3. Ensure code passes linting (`ruff check .`) and tests (`pytest`).
4. Submit a pull request with a clear description and, if applicable, updated documentation.

See `CONTRIBUTING.md` for detailed guidelines.

## License
This project is licensed under the **MIT License**. See the `LICENSE` file for full terms.