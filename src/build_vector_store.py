from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from chunk_documents import chunk_documents

def build_vector_store(output_dir="../vector_store/"):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = chunk_documents()
    langchain_docs = []
    for doc in documents:
        for chunk in doc["chunked_data"]:
            langchain_docs.append(
                Document(
                    page_content=chunk["text"],
                    metadata=chunk["metadata"]
                )
            )
    vector_store = FAISS.from_documents(langchain_docs, embedding_model)
    vector_store.save_local(output_dir)
    print(f"Vector store saved to {output_dir}")

if __name__ == "__main__":
    build_vector_store()