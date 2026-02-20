from dotenv import load_dotenv
load_dotenv()

import os
import shutil
import gradio as gr
from src.ingestion import ingest_file
from src.rag_pipeline import create_rag_chain
from src.utils import format_sources
from src.evaluation import run_evaluation


# ─────────────────────────────────────────────────────────────────────────────
# Upload handler
# ─────────────────────────────────────────────────────────────────────────────

def process_upload(files):
    """
    Handles file upload from the Gradio UI.
        1. Saves the uploaded file(s) to the data/ directory.
        2. Runs the ingestion pipeline (load -> chunk -> embed -> store).
        3. Replaces any previously loaded document entirely.

    NOTE: In Gradio 4+, gr.File returns file paths as strings (not objects).
    """
    if not files:
        return "No file uploaded. Please select a PDF or DOCX file."

    os.makedirs("data", exist_ok=True)

    results = []
    for file_path in files:
        filename = os.path.basename(file_path)
        destination = os.path.join("data", filename)

        try:
            shutil.copy(file_path, destination)
            ingest_file(destination)
            results.append(f"✅ Successfully processed: {filename}")
        except Exception as e:
            results.append(f"❌ Error processing {filename}: {str(e)}")

    return "\n".join(results)


# ─────────────────────────────────────────────────────────────────────────────
# Chat handler
# ─────────────────────────────────────────────────────────────────────────────

def chat_function(message, history):
    """
    Main chat handler for the Gradio ChatInterface.
    Runs the RAG chain and returns the answer with source citations.
    """
    if not message.strip():
        return "Please enter a question."

    try:
        rag_chain = create_rag_chain()
        response = rag_chain.invoke({"input": message})

        answer = response.get("answer", "No answer returned.")
        source_text = format_sources(response.get("context", []))

        if source_text:
            final_answer = f"{answer}\n\n---\n**Sources:**\n{source_text}"
        else:
            final_answer = answer

        return final_answer

    except FileNotFoundError:
        return (
            "No document has been ingested yet. "
            "Please go to the **Upload Contract** tab and upload a PDF or DOCX file first."
        )
    except EnvironmentError as e:
        return f"Configuration error: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation handler
# ─────────────────────────────────────────────────────────────────────────────

def run_eval_handler(num_questions: int):
    """
    Triggered by the Evaluate button in the UI.
    Runs the LLM-as-a-Judge evaluation and returns the markdown report
    and a progress/score summary string.
    """
    try:
        report, score = run_evaluation(num_questions=int(num_questions))
        score_summary = f"### Preference Score: {score:.0%}"
        verdict = "✅ PASS" if score >= 0.5 else "❌ Needs Improvement"
        score_summary += f"  —  {verdict}"
        return score_summary, report
    except FileNotFoundError:
        msg = (
            "❌ No document ingested yet.\n\n"
            "Please upload a PDF or DOCX in the **Upload Contract** tab first."
        )
        return msg, ""
    except EnvironmentError as e:
        return f"❌ Configuration error: {e}", ""
    except Exception as e:
        return f"❌ Evaluation failed: {e}", ""


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI Layout
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(), title="Smart Contract Assistant") as app:

    gr.Markdown("# 📄 Smart Contract Assistant")
    gr.Markdown(
        "Upload a contract (PDF or DOCX), ask questions about it, "
        "and evaluate how well the RAG pipeline understands your document."
    )

    # ── Tab 1: Upload ──────────────────────────────────────────────────────────
    with gr.Tab("1. Upload Contract"):
        gr.Markdown("### Step 1: Upload your document")
        gr.Markdown(
            "Supported formats: **PDF**, **DOCX**. "
            "Uploading a new document will **replace** the previous one."
        )
        upload_component = gr.File(
            label="Select PDF or DOCX",
            file_types=[".pdf", ".docx"],
            file_count="multiple",
        )
        upload_status = gr.Textbox(
            label="Processing Status",
            interactive=False,
            lines=3,
        )
        upload_component.upload(
            fn=process_upload,
            inputs=upload_component,
            outputs=upload_status,
        )

    # ── Tab 2: Chat ────────────────────────────────────────────────────────────
    with gr.Tab("2. Chat with Contract"):
        gr.Markdown("### Step 2: Ask questions about your document")
        gr.Markdown(
            "_Answers are grounded in the uploaded document. "
            "This tool does not provide legal advice._"
        )
        gr.ChatInterface(
            fn=chat_function,
            examples=[
                "What are the key obligations of each party?",
                "What is the termination clause?",
                "Are there any penalty or liability clauses?",
                "Summarise the payment terms.",
            ],
            cache_examples=False,
        )

    # ── Tab 3: Evaluate ────────────────────────────────────────────────────────
    with gr.Tab("3. Evaluate RAG Pipeline"):
        gr.Markdown("### Step 3: Run LLM-as-a-Judge Evaluation")
        gr.Markdown(
            "This tool automatically measures how well the RAG pipeline understands "
            "your uploaded document using a **four-step evaluation loop**:\n\n"
            "1. **Sample** — Randomly pick pairs of chunks from your document\n"
            "2. **Synthesise** — Generate ground-truth Q&A pairs from those chunks\n"
            "3. **Retrieve** — Ask the RAG chain the same questions independently\n"
            "4. **Judge** — An LLM compares both answers and scores `[1]` (worse) or `[2]` (equal/better)\n\n"
            "The final **Preference Score** is the % of questions where the RAG answer "
            "was judged as good or better than the ground truth."
        )

        with gr.Row():
            num_q_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="Number of Evaluation Questions",
                info="More questions = more reliable score, but takes longer to run.",
                scale=3,
            )
            eval_button = gr.Button("▶ Run Evaluation", variant="primary", scale=1)

        score_display = gr.Markdown(
            value="*Score will appear here after running evaluation.*"
        )

        report_display = gr.Markdown(
            label="Full Evaluation Report",
            value=""
        )

        eval_button.click(
            fn=run_eval_handler,
            inputs=[num_q_slider],
            outputs=[score_display, report_display],
        )

        gr.Markdown(
            "---\n"
            "**Interpreting your score:**\n"
            "- **80–100%** — Excellent. The pipeline retrieves and uses context reliably.\n"
            "- **50–79%** — Good. Mostly works, minor retrieval gaps.\n"
            "- **Below 50%** — Consider adjusting chunk size, overlap, or the system prompt.\n\n"
            "*Evaluation uses the same Groq/Llama model as the chat pipeline. "
            "Results are probabilistic — run multiple times for a stable estimate.*"
        )


if __name__ == "__main__":
    app.launch()
