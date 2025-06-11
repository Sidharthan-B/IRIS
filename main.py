from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from rag_hierarchy import prepare_vector_stores, hierarchical_search  # Your new module
from langchain_together import ChatTogether

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class QuestionRequest(BaseModel):
    question: str

# Load LLM and Prompt

#model = OllamaLLM(model="llama3.2")


model = ChatTogether(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",  # You can use "meta-llama/Llama-2-70b-chat" too
    together_api_key="35441b20e7238d32e30897203a9df92aa27c9405771b8eb14a2527b9945ae39f",
    temperature=0.3,
    max_tokens=512
)

template = """
You are a helpful assistant. Answer the user's question using relevant excerpts from the retrieved documents.

Make sure to:
- Provide a clear, concise, and complete answer.
- Avoid repeating the chunks verbatim.
- Format the response nicely using markdown (like bullet points, bold text, or headings if needed).
- Do not mention that the answer was retrieved from documents.
- Clean th answer to just include text, do not add anything else.

Here are some relevant excerpts:
{reviews}

Now, answer this question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Load vector stores and embedders once
doc_store, chunk_store, doc_embedder, chunk_embedder = prepare_vector_stores(reset=False)

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    import time
    start = time.time()

    retrieved_docs = hierarchical_search(
        request.question,
        doc_store,
        chunk_store,
        doc_embedder,
        chunk_embedder,
        k_docs=3,
        k_chunks=5
    )
    after_retrieval = time.time()

    reviews = "\n\n".join([doc.page_content for doc in retrieved_docs])
    result = chain.invoke({"reviews": reviews, "question": request.question})
    raw_response = chain.invoke({"reviews": reviews, "question": request.question})
    clean_answer = raw_response.content if hasattr(raw_response, "content") else str(raw_response)

    after_llm = time.time()

    sources_detailed = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "chunk_id": doc.metadata.get("chunk_id", "?"),
            "text": doc.page_content
        }
        for doc in retrieved_docs
    ]

    return {
        "answer": clean_answer,
        "sources": sources_detailed,
        "timing": {
            "retrieval": round(after_retrieval - start, 2),
            "llm": round(after_llm - after_retrieval, 2)
        }
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
