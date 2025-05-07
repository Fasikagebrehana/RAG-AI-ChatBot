import gradio as gr
from rag_pipeline import load_rag_pipeline, query_rag

def chatbot(query):
    rag_chain = load_rag_pipeline()
    result = query_rag(query, rag_chain)
    answer = result["answer"]
    sources = "\n".join([f"{doc['filename']} (Chunk {doc['chunk_id']})" for doc in result["sources"]])
    return f"**Answer**: {answer}\n\n**Sources**:\n{sources}"

interface = gr.Interface(
    fn=chatbot,
    inputs="text",
    outputs="markdown",
    title="Legal Document Chatbot",
    description="Ask questions about legal documents, and I'll answer based on the provided PDFs. This is not legal advice."
)

if __name__ == "__main__":
    interface.launch()