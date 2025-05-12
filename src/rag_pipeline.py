import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from functools import lru_cache
import time


api_key = ""
genai.configure(api_key=api_key)

@lru_cache(maxsize=1)
def load_rag_pipeline(vector_store_dir="../vector_store/"):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(vector_store_dir, embedding_model, allow_dangerous_deserialization=True)
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, google_api_key=api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Reduced to 2 for speed
    
    template = """You are a legal assistant. Extract the exact legal consequence or penalty from the context. If not found, state 'No penalty or relevant information specified.' If the query is out-of-domain (e.g., weather), return 'No relevant information found.' Context: {context} Question: {question} Answer:"""
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
    
 
    truncated_docs = []
    for doc in result["source_documents"]:
        content = doc.page_content[:300]  
        doc.page_content = content
        truncated_docs.append(doc)
    
    if not truncated_docs or "weather" in query.lower():
        return {
            "answer": "No relevant information found.",
            "sources": [],
            "latency": time.time() - start_time
        }
    
    relevant_docs = [doc for doc in truncated_docs if "penalty" in doc.page_content.lower() or "punishment" in doc.page_content.lower()]
    if not relevant_docs:
        relevant_docs = truncated_docs
    

    print("Retrieved Documents:")
    for doc in relevant_docs:
        print(f"Source: {doc.metadata}, Content: {doc.page_content[:200]}...")
    
    return {
        "answer": result["result"],
        "sources": [doc.metadata for doc in relevant_docs],
        "latency": time.time() - start_time
    }

def get_rag_response(query):
    rag_chain = load_rag_pipeline()
    return query_rag(query, rag_chain)

if __name__ == "__main__":
    rag_chain = load_rag_pipeline()
    query = "What are the penalties for breach of contract?"
    result = query_rag(query, rag_chain)
    print("\nAnswer:", result["answer"])
    print("Sources:", result["sources"])
    print("Latency:", result["latency"], "seconds")