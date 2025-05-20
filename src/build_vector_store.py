from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from chunk_documents import chunk_documents
import time

def build_vector_store(output_dir="../vector_store/", batch_size=50):
    start_time = time.time()
    # Initialize a lighter embedding model to speed up
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Lighter model
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    documents = chunk_documents()
    print(f"Number of documents: {len(documents)}")
    total_chunks = sum(len(doc["chunked_data"]) for doc in documents)
    print(f"Total chunks to process: {total_chunks}")
    
    langchain_docs = []
    for doc in documents:
        for chunk in doc["chunked_data"]:
            langchain_docs.append(
                Document(
                    page_content=chunk["text"],
                    metadata=chunk["metadata"]
                )
            )
    
    # Process in smaller batches
    vector_store = None
    for i in range(0, len(langchain_docs), batch_size):
        batch = langchain_docs[i:i + batch_size]
        batch_start = time.time()
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embedding_model)
        else:
            vector_store.add_documents(batch)
        print(f"Processed batch {i // batch_size + 1}/{len(langchain_docs) // batch_size + 1} "
              f"({len(batch)} chunks) in {time.time() - batch_start:.2f} seconds")
    
    vector_store.save_local(output_dir)
    print(f"Vector store saved to {output_dir} in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    build_vector_store()