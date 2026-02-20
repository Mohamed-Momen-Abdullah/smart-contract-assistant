import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.vector_store import save_to_vectorstore


def load_document(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or DOCX file.")

    return loader.load()


def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} page(s) into {len(chunks)} chunks.")
    return chunks


def ingest_file(file_path):
    print(f"Starting ingestion for: {file_path}")

    raw_docs = load_document(file_path)
    chunks = chunk_documents(raw_docs)
    save_to_vectorstore(chunks)

    print("Ingestion complete. Embeddings stored successfully.")


if __name__ == "__main__":
    # Quick test: place a PDF at data/sample_contract.pdf and run this script directly
    test_pdf = "data/sample_contract.pdf"
    if os.path.exists(test_pdf):
        ingest_file(test_pdf)
    else:
        print(f"Test file not found. Place a PDF at '{test_pdf}' to test ingestion.")
