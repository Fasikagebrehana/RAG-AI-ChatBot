import os

def chunk_text(text, max_words=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks

def chunk_documents(input_dir="../extracted_text/"):
    documents = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
                content = f.read()
            chunks = chunk_text(content)
            chunked_data = [
                {"text": chunk, "metadata": {"filename": filename, "chunk_id": i}}
                for i, chunk in enumerate(chunks)
            ]
            documents.append({"filename": filename, "content": content, "chunked_data": chunked_data})
    return documents

if __name__ == "__main__":
    documents = chunk_documents()
    print(f"Chunked {len(documents)} documents")