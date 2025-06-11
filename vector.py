from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import shutil

# Directory containing text files
data_path = "data/documents"

# Load text files
def load_text_files(directory):
    documents = []
    ids = []
    file_index = 0
    
    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_name.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                paragraphs = text.split("\n\n")  # Splitting on double newlines
                for j, para in enumerate(paragraphs):
                    doc_id = f"{file_index}_{j}"
                    documents.append(Document(
                        page_content=para.strip(),
                        metadata={"source": file_name, "chunk_id": j},
                        id=doc_id
                    ))
                    ids.append(doc_id)
                file_index += 1
    return documents, ids

# Load and process documents
documents, ids = load_text_files(data_path)

# Initialize embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Path to ChromaDB
db_location = "./chroma_langchain_db"

# Delete old ChromaDB directory before starting
if os.path.exists(db_location):
    shutil.rmtree(db_location)
    print("Deleted old ChromaDB.")

# Initialize ChromaDB
vector_store = Chroma(
    collection_name="text_chunks",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Add new documents to ChromaDB
vector_store.add_documents(documents=documents, ids=ids)

# Set up retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

print("Documents successfully processed and stored in ChromaDB.")
