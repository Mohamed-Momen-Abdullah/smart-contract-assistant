import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


VECTOR_STORE_PATH = "faiss_index"

def get_embedding_model():
    """
    Returns the embedding model.
    Using HuggingFace sentence-transformers for local, free embedding.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def save_to_vectorstore(chunks):
    """
    Takes text chunks, creates embeddings, and saves them to the local FAISS index.
    Always creates a FRESH index, replacing any previous document entirely.
    This ensures uploading a new document never mixes data from old documents.
    """
    embeddings = get_embedding_model()

    # Always wipe the old index first so old document data is never retained
    if os.path.exists(VECTOR_STORE_PATH):
        shutil.rmtree(VECTOR_STORE_PATH)
        print("Cleared old vector store.")

    # Build a fresh index from the new document's chunks
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"Created new vector store with {len(chunks)} chunks.")

    # Persist to disk
    vectorstore.save_local(VECTOR_STORE_PATH)
    print(f"Vector store saved to '{VECTOR_STORE_PATH}'.")
    return vectorstore

def load_vectorstore():
    """
    Loads the existing FAISS vector store for retrieval.
    Raises a clear error if ingestion hasn't been run yet.
    """
    embeddings = get_embedding_model()

    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(
            f"Vector store not found at '{VECTOR_STORE_PATH}'. "
            "Please upload and process a document first."
        )

    vectorstore = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True  # for local FAISS files
    )
    return vectorstore