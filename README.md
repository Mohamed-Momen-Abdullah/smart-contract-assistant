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
# Smart Contract Assistant

A RAG-powered web application that allows users to upload legal documents (PDF or DOCX) and interact with them via a conversational AI assistant. Answers are grounded in the uploaded document with source citations.

---

## Features

- Upload PDF or DOCX contracts and legal documents
- Ask natural language questions about the document
- Receive answers with source citations (page references)
- Conversation history within a session
- Legal disclaimer on all responses
- Clean two-tab UI: Upload → Chat

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Gradio |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Embeddings | HuggingFace SentenceTransformers |
| Vector Store | FAISS |
| RAG Framework | LangChain 0.3.x |
| File Parsing | PyPDF, python-docx |

---

## Requirements

- Python 3.11 (required — Python 3.12+ has compatibility issues with LangChain)
- A [Groq API key](https://console.groq.com)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Mohamed-Momen-Abdullah/smart-contract-assistant.git 
cd smart-contract-assistant
```

### 2. Ensure Python 3.11 is installed

```bash
python3.11 --version
```

If not installed, download it from [python.org](https://www.python.org/downloads/).

### 3. Create a virtual environment with Python 3.11

```bash
python3.11 -m venv venv
source venv/bin/activate        # Linux/Mac
# OR
venv\Scripts\activate           # Windows
```

> ⚠️ Always make sure `(venv)` appears in your terminal prompt before running any commands.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Set up environment variables

Create a `.env` file in the project root:

```bash
nano .env
# or
vim .env
```

Edit `.env` and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 6. Run the application

```bash
python app.py
```

Open your browser and go to: **http://127.0.0.1:7860**

---

## Usage

1. Go to the **Upload Contract** tab
2. Upload a PDF or DOCX file
3. Wait for the document to be processed
4. Switch to the **Chat with Contract** tab
5. Ask questions about your document

**Example questions to try:**
- "What are the payment terms?"
- "What is the termination clause?"
- "What are the confidentiality obligations?"
- "Who are the parties involved?"
- "What is the total contract value?"

---

## Dependencies (requirements.txt)

```
langchain==0.3.27
langchain-core==0.3.83
langchain-community==0.3.31
langchain-groq==0.2.5
langchain-huggingface==0.1.2
faiss-cpu
sentence-transformers
python-docx
pypdf
gradio
python-dotenv
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'dotenv'`**
You are running Python directly instead of through the venv. Run `source venv/bin/activate` first.

**`ModuleNotFoundError: No module named 'langchain.chains'`**
Your LangChain version is too new (v1.x). Recreate the venv with Python 3.11 and install pinned versions from `requirements.txt`.

**`model has been decommissioned`**
The Groq model name in your config is outdated. For me, Update it to `llama-3.3-70b-versatile`, and it works fine.

**App loads but answers are wrong**
Ensure the document was fully processed before switching to the chat tab. Try re-uploading the file.

---

## Notes

- This tool is for informational purposes only and does not constitute legal advice.
- All processing is done locally — documents are not sent to any external storage.
- Session data is cleared when the app restarts.

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
