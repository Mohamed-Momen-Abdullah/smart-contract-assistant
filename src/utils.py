import os


def format_sources(context_docs: list) -> str:
    """
    Takes a list of LangChain Document objects (from the RAG chain's
    'context' key) and returns a formatted string of unique source citations.

    Each citation shows the filename and page number where available.

    Args:
        context_docs: List of Document objects with metadata.

    Returns:
        A newline-separated string of source citations, or an empty string
        if no valid sources are found.
    """
    if not context_docs:
        return ""

    seen = set()
    citations = []

    for doc in context_docs:
        source_file = os.path.basename(doc.metadata.get("source", "Unknown document"))
        page = doc.metadata.get("page", None)

        if page is not None:
            # Page numbers from PyPDFLoader are 0-indexed; add 1 for human-readable display
            citation = f"- {source_file}, Page {int(page) + 1}"
        else:
            citation = f"- {source_file}"

        if citation not in seen:
            seen.add(citation)
            citations.append(citation)

    return "\n".join(citations)
