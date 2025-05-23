import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from functools import lru_cache
import time
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")
genai.configure(api_key=api_key)

@lru_cache(maxsize=1)
def load_rag_pipeline(vector_store_dir=None):
    if vector_store_dir is None:
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        vector_store_dir = os.path.join(current_dir, "vector_store")
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # Explicitly set to CPU
    )
    vector_store = FAISS.load_local(vector_store_dir, embedding_model, allow_dangerous_deserialization=True)
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5, google_api_key=api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    template = """You are a legal assistant. Based on the context, provide a detailed response to the question about legal rights or provisions (e.g., property division, constitutional protections). Use specific article references if available. If no relevant information is found, explain the limitation and suggest consulting a legal expert. Context: {context} Question: {question} Answer:"""
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return rag_chain

def query_rag(query, rag_chain):
    start_time = time.time()
    result = rag_chain.invoke({"query": query})
    answer = result["result"]
    truncated_docs = [doc for doc in result["source_documents"]]
    
    if not truncated_docs or "weather" in query.lower():
        return {
            "answer": "No relevant information found.",
            "sources": [],
            "source_previews": [],
            "latency": time.time() - start_time
        }
    
    relevant_docs = [doc for doc in truncated_docs if any(keyword in doc.page_content.lower() for keyword in ["property", "divorce", "rights"])]
    if not relevant_docs:
        relevant_docs = truncated_docs
    
    print("Retrieved Documents:")
    for doc in relevant_docs:
        print(f"Source: {doc.metadata}, Content: {doc.page_content[:200]}...")
    
    return {
        "answer": answer,
        "sources": [doc.metadata for doc in relevant_docs],
        "source_previews": [{"filename": doc.metadata["filename"], "chunk_id": doc.metadata["chunk_id"], "text": doc.page_content[:200]} for doc in relevant_docs],
        "latency": time.time() - start_time
    }

if __name__ == "__main__":
    rag_chain = load_rag_pipeline()
    query = "What are my rights to property division under the Family Code?"
    result = query_rag(query, rag_chain)
    print("\nAnswer:", result["answer"])
    print("Sources:", result["sources"])
    print("Latency:", result["latency"], "seconds")