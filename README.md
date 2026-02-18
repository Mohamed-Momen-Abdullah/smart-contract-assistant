# Smart Contract Assistant

A Retrieval-Augmented Generation (RAG) web application that lets you upload PDF or DOCX contracts and ask questions about them via a conversational chat interface.

Built with LangChain, FAISS, Groq (Llama 3 70B), HuggingFace Embeddings, and Gradio.

---

## Project Structure

```
smart-contract-assistant/
├── data/                 # Uploaded documents are saved here (auto-created)
├── faiss_index/          # FAISS vector store (auto-created on first upload)
├── src/
│   ├── __init__.py       # Makes src a Python package
│   ├── ingestion.py      # Load → Chunk → Embed pipeline
│   ├── vector_store.py   # FAISS save/load logic
│   ├── rag_pipeline.py   # LLM chain with guardrails
│   └── utils.py          # Citation formatting helper
├── app.py                # Gradio UI entry point
├── requirements.txt      # Python dependencies
├── .env                  # API keys (never commit this file)
└── README.md
```

---

## Setup

### 1. Clone the repository and install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create your `.env` file

Create a file named `.env` in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at https://console.groq.com

### 3. Run the application

```bash
python app.py
```

The Gradio UI will open at `http://127.0.0.1:7860`.

---

## Usage

1. **Upload Contract tab** — Upload a PDF or DOCX contract. The system will parse, chunk, embed, and store it in the local FAISS vector store.
2. **Chat with Contract tab** — Ask any question about the document. The assistant will retrieve the most relevant sections and generate a grounded answer with source page citations.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| FAISS (local) | No external service needed; fast and free |
| `all-MiniLM-L6-v2` embeddings | Lightweight, runs locally, no API key required |
| Groq + Llama 3 70B | Free tier, very fast inference |
| Chunk size 1000 / overlap 200 | Balances context richness with retrieval precision |
| Temperature 0 | Deterministic outputs for factual contract analysis |

---

## Guardrails

- The LLM is instructed to answer **only** from retrieved context.
- If the answer is not in the document, it will say so explicitly.
- All answers involving legal obligations include a disclaimer.

---

## Limitations

- English-language documents only.
- Single-session; conversation history is not forwarded to the LLM.
- Not suitable for production deployment without additional security hardening.
