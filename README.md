# 🧠 IRIS – Institutional Retrieval-based Information System

**IRIS** is a Hierarchical Retrieval-Augmented Generation (RAG) application designed to answer institution-specific questions using uploaded textual documents. By combining semantic document search with large language models, IRIS delivers **context-aware, accurate, and concise responses** based on relevant institutional knowledge.

---

## 🚀 Key Features

- 🔍 **Hierarchical Retrieval**  
  First filters relevant documents, then narrows down to the most relevant chunks.
  
- 🤖 **LLM-Powered Responses**  
  Uses TogetherAI-hosted models (e.g., Mixtral, LLaMA) for generating fluent, high-quality answers.

- 📁 **Custom Document Support**  
  Simply place `.txt` files in `data/documents/` and they’re ready for querying.

- 🌐 **Web-based UI via Gradio**  
  Easy-to-use interface to ask questions, preview documents, and see retrieval metadata.

- ⚙️ **Built with FastAPI, LangChain, ChromaDB, Ollama Embeddings, and Gradio**

---

## 📂 Project Structure

iris/
├── gradio_app.py # Gradio frontend
├── main.py # FastAPI backend for retrieval + LLM answer generation
├── rag_hierarchy.py # Core RAG logic: document loading, vector store creation, search
├── vector.py # Standalone script to rebuild ChromaDB
├── requirements.txt # Python dependencies
├── data/
│ └── documents/ # Place your .txt files here
└── README.md # This file



---

## ⚙️ Setup Instructions

### 1. Clone the Repo

git clone https://github.com/yourusername/iris.git
cd iris

2. Set Up Virtual Environment
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate

3. Install Dependencies
   pip install -r requirements.txt

4. Add Documents
   Put your .txt files inside data/documents/.


🧠 Backend: FastAPI (LLM & Search)

Start the backend API:  python main.py

This starts the IRIS API at http://localhost:5000/ask.

It:
   Performs hierarchical retrieval (document → chunk)
   Generates answers using a TogetherAI-hosted LLM
   Returns source chunks and timing stats



💻 Frontend: Gradio Interface

Start the frontend:  python gradio_app.py

This launches a local Gradio UI:
                                 Ask questions
                                 View answers and response time
                                 Preview uploaded documents
                                 See which document chunks were retrieved


🔐 Environment Variables (Optional)

You can store your Together API key and base URL in a .env file:  TOGETHER_API_KEY=your_api_key_here
                                                                  API_URL=http://localhost:5000/ask




📌 Notes


IRIS uses Ollama Embeddings (mxbai-embed-large, nomic-embed-text) for encoding text and ChromaDB for vector storage.

The retrieval pipeline first selects top documents, then finds best-matching chunks within them.

The LLM is instructed to not quote the documents directly, but synthesize a clean, natural-language answer.

Full markdown formatting supported in answers.







  




