from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.rag_pipeline import load_rag_pipeline, query_rag

app = FastAPI(title="Legal Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend.vercel.app"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the RAG pipeline once at startup
rag_chain = load_rag_pipeline()

@app.get("/")
async def root():
    return {"message": "Welcome to the Legal Chatbot API. Use POST /chat to interact with the chatbot."}

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: ChatRequest):
    print(f"Received query: {request.query}")
    try:
        result = query_rag(request.query, rag_chain)
        print(f"Query result: {result}")
        return result
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)