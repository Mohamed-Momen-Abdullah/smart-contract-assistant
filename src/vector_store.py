import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


VECTOR_STORE_PATH = "faiss_index"

def get_embedding_model():

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def save_to_vectorstore(chunks):
    
    embeddings = get_embedding_model()

    # wipe the old index first so old document data is never in my way
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