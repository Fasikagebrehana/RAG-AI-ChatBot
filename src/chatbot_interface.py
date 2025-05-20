import gradio as gr
from rag_pipeline import load_rag_pipeline, query_rag
from fpdf import FPDF
import time

rag_chain = load_rag_pipeline()

def export_to_pdf(query, response):
    if response == "Please enter a legal question.":
        return None
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Lawfind Response\n\nQuery: {query}\n\n{response}")
    output_file = f"lawfind_response_{int(time.time())}.pdf"
    pdf.output(output_file)
    return output_file

def respond(query):
    if query:
        result = query_rag(query, rag_chain)
        answer = result["answer"]
        sources = result["sources"]
        previews = "\n".join([f"- {p['filename']} (Chunk {p['chunk_id']}): {p['text']}..." for p in result.get("source_previews", [])])
        response = f"Answer:\n{answer}\n\nSources:\n{', '.join([f'{s['filename']} (Chunk {s['chunk_id']})' for s in sources])}\n\nSource Previews:\n{previews if previews else 'None'}"
        metrics = f"Latency: {result['latency']:.2f}s | Sources Retrieved: {len(sources)}"
        return response, metrics, query
    return "Please enter a legal question.", "", query

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="Lawfind") as interface:
    gr.Markdown("""
    <h1 style='text-align: center; color: #1E3A8A'>Lawfind: Your Legal Document Assistant</h1>
    <p style='text-align: center'>Ask questions about legal documents (e.g., Family Code, Constitution). Not legal advice.</p>
    """)
    with gr.Row():
        query_input = gr.Textbox(label="Ask a Legal Question", placeholder="e.g., What are my rights under the Family Code?", lines=2)
        submit_btn = gr.Button("Submit", variant="primary")
        clear_btn = gr.Button("Clear", variant="secondary")
    output = gr.Markdown(label="Answer and Sources")
    metrics_output = gr.Textbox(label="Performance Metrics", interactive=False)
    with gr.Row():
        export_btn = gr.Button("Export as PDF", variant="secondary")
        export_output = gr.File(label="Download Response")
    
    submit_btn.click(fn=respond, inputs=[query_input], outputs=[output, metrics_output, gr.State()])
    query_input.submit(fn=respond, inputs=[query_input], outputs=[output, metrics_output, gr.State()])
    clear_btn.click(fn=lambda: ("", "", ""), outputs=[query_input, output, metrics_output])
    export_btn.click(fn=export_to_pdf, inputs=[query_input, output], outputs=[export_output])

interface.launch(server_name="0.0.0.0", server_port=7860)