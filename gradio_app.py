import gradio as gr
import requests
import os
import time

API_URL = os.getenv("API_URL", "http://localhost:5000/ask")
DATA_DIR = "data/documents"

def ask_bot(question):
    if not question.strip():
        return "Please enter a question.", "", [], gr.update(visible=False)

    try:
        response = requests.post(API_URL, json={"question": question})
        data = response.json()

        answer = data.get("answer", "No answer returned.")
        sources = data.get("sources", [])
        timing = data.get("timing", {"retrieval": "?", "llm": "?"})

        time_stats = f"⏱️ Retrieval: {timing['retrieval']}s | 🤖 LLM: {timing['llm']}s"

        return answer, time_stats, sources, gr.update(visible=True)

    except Exception as e:
        return f"❌ Error contacting backend: {e}", "", [], gr.update(visible=False)

def list_uploaded_files():
    return [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".txt")]

def preview_file(file_path):
    if file_path and os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return "File not found."

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Institution Chatbot\nAsk any question based on the uploaded documents.")

    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(label="Your Question", placeholder="e.g. What is the sliding window protocol?")
            submit_btn = gr.Button("Ask")
            answer_output = gr.Textbox(label="📜 Bot Answer", lines=4, interactive=False)
            time_display = gr.Textbox(label="⏱️ Time Breakdown", interactive=False)

        with gr.Column(scale=2) as chunk_column:
            retrieved_chunks = gr.Textbox(label="🔍 Retrieved Chunks", lines=20, interactive=False)

    def update_ui(question):
        answer, time_info, sources, _ = ask_bot(question)
        chunks_text = "\n\n".join(
            [f"[{s['source']} | chunk {s['chunk_id']}]\n{s['text']}" for s in sources]
        )
        return answer, time_info, chunks_text

    submit_btn.click(
        fn=update_ui,
        inputs=question_input,
        outputs=[answer_output, time_display, retrieved_chunks]
    )

    gr.Markdown("---")
    gr.Markdown("### 📁 Uploaded Documents")

    with gr.Row():
        file_dropdown = gr.Dropdown(label="Select a document", choices=list_uploaded_files())
        preview_box = gr.Textbox(label="📖 File Preview", lines=10, interactive=False)

    file_dropdown.change(fn=preview_file, inputs=file_dropdown, outputs=preview_box)

if __name__ == "__main__":
    demo.launch()
