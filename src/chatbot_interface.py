import gradio as gr
from rag_pipeline import load_rag_pipeline, query_rag
import json
import os

HISTORY_FILE = "chat_history.json"
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        chat_history = json.load(f)
else:
    chat_history = []

rag_chain = load_rag_pipeline()

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

def add_chat_message(query, history):
    if query:
        result = query_rag(query, rag_chain)
        answer = result["answer"]
        sources = result["sources"]
        response = f"{answer}\n\n**Sources:** {', '.join([f'{s['filename']} (Chunk {s['chunk_id']})' for s in sources])}"
        history.append({"query": query, "response": response, "latency": result["latency"]})
        save_history(history)
    return history, response or "Please enter a question."

def start_new_chat(history):
    if history:
        save_history(history)
    return [], ""

def load_chat(chat_id, history):
    if 0 <= chat_id < len(history):
        return history, history[chat_id]["response"]
    return history, "Invalid chat selected."

with gr.Blocks(title="Lawfind") as interface:
    with gr.Row():
        with gr.Column(scale=1, min_width=100):
            gr.Markdown("### Lawfind")
            with gr.Accordion("Recent", open=True):
                chat_list = gr.State(chat_history)
                chat_display = gr.Markdown()
                chat_select = gr.Dropdown(
                    choices=[f"Chat {i+1}: {h['query'][:30]}..." for i, h in enumerate(chat_history)] if chat_history else [],
                    label="Select a previous chat",
                    value=None
                )
                
                def update_chat_display_and_select(history):
                    display_text = "\n".join([f"- {h['query']}" for h in history[-5:]]) if history else "No recent chats."
                    choices = [f"Chat {i+1}: {h['query'][:30]}..." for i, h in enumerate(history)] if history else []
                    return display_text, gr.Dropdown(choices=choices, value=None)
                
                chat_display.value = update_chat_display_and_select(chat_history)[0]
                chat_list.change(
                    fn=update_chat_display_and_select,
                    inputs=[chat_list],
                    outputs=[chat_display, chat_select]
                )

            with gr.Row():
                new_chat_btn = gr.Button("New Chat", variant="secondary")
                new_chat_btn.click(fn=start_new_chat, inputs=[chat_list], outputs=[chat_list, gr.State(value="")])

        with gr.Column(scale=4):
            gr.Markdown("Ask questions about legal documents, and I'll answer based on the provided PDFs. This is not legal advice.")
            with gr.Row():
                query_input = gr.Textbox(label="What can I help you with?", placeholder="Ask a question", lines=1)
                submit_btn = gr.Button("Submit", variant="primary")
            output = gr.Markdown()
            
            def respond(query, history):
                history, response = add_chat_message(query, history)
                return history, response

            submit_btn.click(fn=respond, inputs=[query_input, chat_list], outputs=[chat_list, output])
            query_input.submit(fn=respond, inputs=[query_input, chat_list], outputs=[chat_list, output])
            chat_select.change(fn=load_chat, inputs=[chat_select, chat_list], outputs=[chat_list, output])

interface.launch(server_name="0.0.0.0", server_port=7860)