from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os, shutil

def load_documents(directory):
    chunk_docs, doc_docs, ids = [], [], []
    file_index = 0

    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".txt"):
                with open(os.path.join(root, file_name), "r", encoding="utf-8") as f:
                    text = f.read()

                doc_id = f"{file_index}_doc"
                doc_docs.append(Document(
                    page_content=text.strip(),
                    metadata={"source": file_name},
                    id=doc_id
                ))

                paragraphs = text.split("\n\n")
                for j, para in enumerate(paragraphs):
                    chunk_id = f"{file_index}_{j}"
                    chunk_docs.append(Document(
                        page_content=para.strip(),
                        metadata={"source": file_name, "chunk_id": j},
                        id=chunk_id
                    ))
                    ids.append(chunk_id)
                file_index += 1
    return doc_docs, chunk_docs, ids


def prepare_vector_stores(
    data_dir="data/documents",
    doc_db_path="./chroma_doc_db",
    chunk_db_path="./chroma_chunk_db",
    reset=False
):
    chunk_embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    doc_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    if reset:
        for path in [chunk_db_path, doc_db_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
                print(f"🧹 Deleted old DB at {path}")

    doc_docs, chunk_docs, chunk_ids = load_documents(data_dir)

    doc_vector_store = Chroma(
        collection_name="full_docs",
        persist_directory=doc_db_path,
        embedding_function=doc_embeddings
    )
    chunk_vector_store = Chroma(
        collection_name="text_chunks",
        persist_directory=chunk_db_path,
        embedding_function=chunk_embeddings
    )

    if reset:
        doc_vector_store.add_documents(doc_docs)
        chunk_vector_store.add_documents(documents=chunk_docs, ids=chunk_ids)
        print("✅ Vector stores populated.")
    else:
        print("✅ Using existing vector stores.")

    return doc_vector_store, chunk_vector_store, doc_embeddings, chunk_embeddings


def hierarchical_search(query, doc_store, chunk_store, doc_embedder, chunk_embedder, k_docs=3, k_chunks=5):
    doc_query_embedding = doc_embedder.embed_query(query)
    top_docs = doc_store.similarity_search_by_vector(doc_query_embedding, k=k_docs)
    relevant_sources = [doc.metadata["source"] for doc in top_docs]

    if not relevant_sources:
        print("[hierarchical_search] ❌ No relevant documents found.")
        return []

    chunk_query_embedding = chunk_embedder.embed_query(query)
    filtered_chunks = chunk_store.similarity_search_by_vector(
        chunk_query_embedding,
        k=k_chunks,
        filter={"source": {"$in": relevant_sources}}
    )

    return filtered_chunks


# -------------------------- RUN FOR DEBUGGING --------------------------
if __name__ == "__main__":
    print("🔧 Running vector store setup...")
    doc_vs, chunk_vs, doc_embed, chunk_embed = prepare_vector_stores(reset=True)

    print("\n🔍 Running test query:")
    query = input("Enter your query: ").strip()
    if query:
        results = hierarchical_search(query, doc_vs, chunk_vs, doc_embed, chunk_embed)

        if results:
            print(f"\n✅ Found {len(results)} relevant chunks:\n")
            for res in results:
                print(f"📁 {res.metadata['source']} (chunk {res.metadata['chunk_id']})")
                print(f"→ {res.page_content[:200]}...\n")  # Print first 200 chars
        else:
            print("❌ No chunks found for the given query.")
    else:
        print("⚠️ No query entered.")
