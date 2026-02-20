"""
evaluation.py — LLM-as-a-Judge Evaluation for the Smart Contract Assistant
===========================================================================

This module implements a pairwise RAG evaluation pipeline inspired by the
NVIDIA DLI course's LLM-as-a-Judge formulation. It works in four steps:

  1. SAMPLE    — Pick random chunk pairs from the live FAISS vector store
  2. SYNTHESISE — Ask the LLM to generate a Q&A pair from those chunks (ground truth)
  3. RETRIEVE  — Run the actual RAG chain on the same question
  4. JUDGE     — A second LLM call compares both answers and scores [1] or [2]

The final "Preference Score" is the fraction of questions where the RAG
chain's answer was judged equal to or better than the ground-truth answer.

Usage (standalone):
    python -m src.evaluation          # runs 3 questions, prints a report

Usage (from app.py / Gradio tab):
    from src.evaluation import run_evaluation
    report, score = run_evaluation(num_questions=5)
"""

import random
import re
from typing import Tuple

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.vector_store import load_vectorstore
from src.rag_pipeline import create_rag_chain


# ─────────────────────────────────────────────────────────────────────────────
# LLM helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_judge_llm() -> ChatGroq:
    """
    Returns a Groq LLM for the judge role.
    Using temperature=0 so judgements are deterministic and reproducible.
    """
    import os
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set. Add it to your .env file.")
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=api_key)


def _get_synth_llm() -> ChatGroq:
    """
    Returns a Groq LLM for synthetic Q&A generation.
    Slight temperature helps produce varied, interesting questions.
    """
    import os
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set. Add it to your .env file.")
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4, api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Sample chunks from the live FAISS vector store
# ─────────────────────────────────────────────────────────────────────────────

def _sample_chunks(n: int = 2) -> list:
    """
    Loads the FAISS vector store and randomly samples n document chunks.
    Raises FileNotFoundError if no document has been ingested yet.
    """
    vectorstore = load_vectorstore()
    all_docs = list(vectorstore.docstore._dict.values())

    if len(all_docs) < 2:
        raise ValueError(
            "The vector store contains fewer than 2 chunks. "
            "Please upload a longer document before running evaluation."
        )

    return random.sample(all_docs, min(n, len(all_docs)))


def _format_chunk(doc) -> str:
    """
    Formats a LangChain Document into a readable string for the LLM.
    Includes source filename and page number when available.
    """
    import os
    source = os.path.basename(doc.metadata.get("source", "Unknown"))
    page = doc.metadata.get("page", None)
    page_info = f", Page {int(page) + 1}" if page is not None else ""
    return (
        f"Source: {source}{page_info}\n"
        f"Content: {doc.page_content}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Synthesise a ground-truth Q&A pair from sampled chunks
# ─────────────────────────────────────────────────────────────────────────────

_SYNTH_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert at generating evaluation data for contract analysis systems. "
        "Given two excerpts from a legal/business document, generate ONE clear question "
        "that can be answered using those excerpts, and the correct answer derived strictly "
        "from the text.\n\n"
        "FORMAT (follow exactly, no extra text before or after):\n"
        "Question: <your question here>\n\n"
        "Answer: <your answer here>"
    ),
    (
        "human",
        "Excerpt 1:\n{chunk1}\n\nExcerpt 2:\n{chunk2}"
    ),
])


def _generate_synthetic_qa(chunk1, chunk2) -> Tuple[str, str]:
    """
    Uses the LLM to generate a synthetic question and ground-truth answer
    from two document chunks. Returns (question, answer).
    """
    llm = _get_synth_llm()
    chain = _SYNTH_PROMPT | llm | StrOutputParser()
    output = chain.invoke({
        "chunk1": _format_chunk(chunk1),
        "chunk2": _format_chunk(chunk2),
    })

    # Parse the structured output
    question, answer = "", ""
    parts = output.split("\n\n", 1)
    if len(parts) == 2:
        question = parts[0].replace("Question:", "").strip()
        answer = parts[1].replace("Answer:", "").strip()
    else:
        # Fallback: try line-by-line
        for line in output.splitlines():
            if line.startswith("Question:"):
                question = line.replace("Question:", "").strip()
            elif line.startswith("Answer:"):
                answer = line.replace("Answer:", "").strip()

    return question, answer


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Run the RAG chain on the synthetic question
# ─────────────────────────────────────────────────────────────────────────────

def _get_rag_answer(question: str) -> str:
    """
    Runs the full RAG pipeline on a given question and returns the answer string.
    """
    rag_chain = create_rag_chain()
    response = rag_chain.invoke({"input": question})
    return response.get("answer", "No answer returned.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Judge: compare synthetic answer vs RAG answer
# ─────────────────────────────────────────────────────────────────────────────

_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an impartial judge evaluating the quality of AI-generated answers "
        "to questions about legal and contract documents.\n\n"
        "You will be shown:\n"
        "  - A question\n"
        "  - Answer 1: a ground-truth answer derived directly from document excerpts\n"
        "  - Answer 2: an answer produced by a RAG (retrieval-augmented generation) system\n\n"
        "SCORING RULES:\n"
        "[1] = Answer 2 is WORSE than Answer 1. It misses key information, "
        "      is factually inconsistent, fabricates details, or fails to answer the question.\n"
        "[2] = Answer 2 is EQUAL TO or BETTER THAN Answer 1. It correctly answers "
        "      the question without introducing errors or hallucinations.\n\n"
        "Output format (follow exactly):\n"
        "[Score] Brief justification (1-2 sentences)"
    ),
    (
        "human",
        "Question: {question}\n\n"
        "Answer 1 (Ground Truth): {synth_answer}\n\n"
        "Answer 2 (RAG Answer): {rag_answer}\n\n"
        "EVALUATION:"
    ),
])


def _judge_pair(question: str, synth_answer: str, rag_answer: str) -> Tuple[int, str]:
    """
    Asks the judge LLM to compare the synthetic answer (ground truth) against
    the RAG answer. Returns (score, justification) where score is 1 or 2.
    """
    llm = _get_judge_llm()
    chain = _JUDGE_PROMPT | llm | StrOutputParser()
    result = chain.invoke({
        "question": question,
        "synth_answer": synth_answer,
        "rag_answer": rag_answer,
    })

    score = 2 if "[2]" in result else 1
    justification = re.sub(r"^\[.\]\s*", "", result).strip()
    return score, justification


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(num_questions: int = 3) -> Tuple[str, float]:
    """
    Runs the full LLM-as-a-Judge evaluation pipeline.

    Args:
        num_questions: How many synthetic Q&A pairs to evaluate (default: 3).

    Returns:
        report (str): A human-readable markdown report of the full evaluation.
        score  (float): The preference score — fraction of [2] judgements (0.0–1.0).
    """
    results = []

    for i in range(num_questions):
        # ── Step 1: Sample two random chunks ──────────────────────────────────
        try:
            chunks = _sample_chunks(2)
        except (FileNotFoundError, ValueError) as e:
            return f"❌ Evaluation failed: {e}", 0.0

        chunk1, chunk2 = chunks[0], chunks[1]

        # ── Step 2: Generate synthetic ground-truth Q&A ───────────────────────
        try:
            question, synth_answer = _generate_synthetic_qa(chunk1, chunk2)
        except Exception as e:
            question, synth_answer = f"[Generation failed: {e}]", ""

        # ── Step 3: Get RAG chain answer ───────────────────────────────────────
        try:
            rag_answer = _get_rag_answer(question)
        except Exception as e:
            rag_answer = f"[RAG chain error: {e}]"

        # ── Step 4: Judge the pair ─────────────────────────────────────────────
        try:
            score, justification = _judge_pair(question, synth_answer, rag_answer)
        except Exception as e:
            score, justification = 1, f"[Judge error: {e}]"

        results.append({
            "question": question,
            "synth_answer": synth_answer,
            "rag_answer": rag_answer,
            "score": score,
            "justification": justification,
        })

    # ── Aggregate and build report ─────────────────────────────────────────────
    preference_score = sum(r["score"] == 2 for r in results) / len(results)
    report = _build_report(results, preference_score)
    return report, preference_score


def _build_report(results: list, preference_score: float) -> str:
    """
    Formats the evaluation results into a readable markdown report.
    """
    passed = sum(r["score"] == 2 for r in results)
    total = len(results)

    verdict = "✅ PASS" if preference_score >= 0.5 else "❌ NEEDS IMPROVEMENT"

    lines = [
        f"# RAG Evaluation Report",
        f"",
        f"**Overall Preference Score:** {preference_score:.0%}  ({passed}/{total} questions passed)",
        f"**Verdict:** {verdict}",
        f"",
        "---",
        "",
        "> **How to read this:** For each question, a synthetic ground-truth answer was generated "
        "directly from document excerpts. The RAG chain then answered the same question independently. "
        "A judge LLM compared both answers: **[2] = RAG answer is as good or better**, **[1] = RAG answer is worse**.",
        "",
        "---",
        "",
    ]

    for i, r in enumerate(results, 1):
        badge = "🟢 [2] Pass" if r["score"] == 2 else "🔴 [1] Fail"
        lines += [
            f"## Question {i} — {badge}",
            "",
            f"**❓ Question:**  ",
            f"{r['question']}",
            "",
            f"**📄 Ground-Truth Answer** *(derived from document excerpts)*:  ",
            f"{r['synth_answer']}",
            "",
            f"**🤖 RAG Chain Answer:**  ",
            f"{r['rag_answer']}",
            "",
            f"**⚖️ Judge's Reasoning:**  ",
            f"{r['justification']}",
            "",
            "---",
            "",
        ]

    lines += [
        "## Interpretation Guide",
        "",
        "| Score | Meaning |",
        "|-------|---------|",
        "| 80–100% | Excellent — RAG chain consistently retrieves and uses context well |",
        "| 50–79%  | Good — RAG chain mostly works; some retrieval gaps may exist |",
        "| Below 50% | Needs work — consider adjusting chunk size, overlap, or prompt |",
        "",
        "*This is an AI-generated evaluation. Results are probabilistic and should be "
        "reviewed alongside manual inspection for production use.*",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("Running evaluation with 3 questions...\n")
    report, score = run_evaluation(num_questions=3)
    print(report)
    print(f"\nFinal Preference Score: {score:.0%}")
