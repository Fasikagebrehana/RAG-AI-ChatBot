from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Characters, adjustable
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]  # Prioritize paragraph/section breaks
    )
    return text_splitter.split_text(text)

def chunk_documents(input_dir="../extracted_text/"):
    documents = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
                content = f.read()
            chunks = chunk_text(content)
            chunked_data = [
                {
                    "text": chunk,
                    "metadata": {
                        "filename": filename,
                        "chunk_id": i,
                        "keywords": ["divorce", "property", "rights"] if "family" in filename.lower() else ["rights"]  # Example keywords
                    }
                }
                for i, chunk in enumerate(chunks)
            ]
            documents.append({"filename": filename, "content": content, "chunked_data": chunked_data})
    return documents

if __name__ == "__main__":
    documents = chunk_documents()
    print(f"Chunked {len(documents)} documents")