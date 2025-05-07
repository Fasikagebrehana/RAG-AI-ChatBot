import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
import google.generativeai as genai

api_key = "AIzaSyDw4Zb6iD3plEzq6f22MKrR1n1XD1LutnQ"
genai.configure(api_key=api_key)

def load_rag_pipeline(vector_store_dir="../vector_store/"):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(vector_store_dir, embedding_model, allow_dangerous_deserialization=True)
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, google_api_key=api_key)  # Explicit API key
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return rag_chain

def query_rag(query, rag_chain):
    result = rag_chain.invoke({"query": query})
    return {
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]]
    }

if __name__ == "__main__":
    rag_chain = load_rag_pipeline()
    query = "What are the penalties for breach of contract?"
    result = query_rag(query, rag_chain)
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])