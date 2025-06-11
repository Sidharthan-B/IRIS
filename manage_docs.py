# manage_docs.py
from rag_hierarchy import prepare_vector_stores

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild vector stores")
    parser.add_argument("--reset", action="store_true", help="Reset and re-embed all documents")

    args = parser.parse_args()

    print("📁 Loading vector stores...")
    prepare_vector_stores(reset=args.reset)
    print("✅ Done!")
