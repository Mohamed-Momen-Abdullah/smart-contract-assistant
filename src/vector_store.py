import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Define where the database will be saved locally
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
    If an index already exists, it merges (adds) the new chunks to it.
    """
    embeddings = get_embedding_model()

    if os.path.exists(VECTOR_STORE_PATH):
        # Load existing index and add new documents
        vectorstore = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
        print(f"Added {len(chunks)} new chunks to existing vector store.")
    else:
        # Create a brand new index
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print(f"Created new vector store with {len(chunks)} chunks.")

    # Save (overwrite) to disk
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
        allow_dangerous_deserialization=True  # Required for local FAISS files
    )
    return vectorstore
