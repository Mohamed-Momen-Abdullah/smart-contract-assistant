from dotenv import load_dotenv
load_dotenv()

import os
import shutil
import gradio as gr
from src.ingestion import ingest_file
from src.rag_pipeline import create_rag_chain
from src.utils import format_sources
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
        # In Gradio 4+, each item is already a string path to a temp file
        filename = os.path.basename(file_path)
        destination = os.path.join("data", filename)

        try:
            shutil.copy(file_path, destination)
            ingest_file(destination)
            results.append(f"Successfully processed: {filename}")
        except Exception as e:
            results.append(f"Error processing {filename}: {str(e)}")

    return "\n".join(results)


def chat_function(message, history):
    """
    Main chat handler for the Gradio ChatInterface.
    Runs the RAG chain and returns the answer with source citations.

    The RAG chain is recreated on every call so it always uses
    the latest uploaded document's vector store.
    """
    if not message.strip():
        return "Please enter a question."

    try:
        rag_chain = create_rag_chain()
        response = rag_chain.invoke({"input": message})

        answer = response.get("answer", "No answer returned.")

        # Append source citations
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


# ---------------------------------------------------------
# Gradio UI Layout
# ---------------------------------------------------------

with gr.Blocks(theme=gr.themes.Soft(), title="Smart Contract Assistant") as app:

    gr.Markdown("# Smart Contract Assistant")
    gr.Markdown(
        "Upload a contract (PDF or DOCX) and ask questions about it. "
        "The assistant will answer using only the content of your document."
    )

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


if __name__ == "__main__":
    app.launch()