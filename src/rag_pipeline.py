import os
from langchain.chains import create_retrieval_chain  # works in v0.3
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from src.vector_store import load_vectorstore

# Llama 3 70B via Groq 
LLM_MODEL = "llama-3.3-70b-versatile"


def get_llm():
    """
    Initialises the Groq LLM. Reads the API key from the environment.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Please add it to your .env file."
        )

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0,     
        api_key=api_key,
    )
    return llm


def get_retriever():
    """
    Loads the FAISS vector store and returns a retriever that fetches
    the top-3 most relevant chunks for a given query.
    """
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def create_rag_chain():
    """
    Builds the full RAG chain:
        Retriever -> StuffDocuments -> LLM
    Includes a system prompt with guardrails to:
        - Ground answers in the retrieved context only
        - Refuse to answer if the answer is not in the document
        - Add a legal disclaimer
    """
    llm = get_llm()
    retriever = get_retriever()

    system_prompt = (
        "You are a professional AI assistant specialising in contract and legal "
        "document analysis. Your job is to help users understand the contents of "
        "the document they have uploaded.\n\n"
        "RULES YOU MUST FOLLOW:\n"
        "1. Base your answer ONLY on the retrieved context provided below. "
        "   Do NOT use outside knowledge.\n"
        "2. If the answer cannot be found in the context, respond with: "
        "   'I could not find information about that in the uploaded document.'\n"
        "3. Never fabricate clauses, dates, names, or obligations.\n"
        "4. Always end answers that involve legal obligations or rights with: "
        "   '*Disclaimer: This is an AI-generated summary for informational "
        "   purposes only and does not constitute legal advice.*'\n\n"
        "Retrieved context from the document:\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain
