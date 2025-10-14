from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

# =====================================================
# CONFIG
# =====================================================
DATA_PATH = "DataUsed/"
DB_FAISS_PATH = "vectorstores/db_faiss"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents(data_path):
    """Load PDF documents from the given folder."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory '{data_path}' does not exist.")

    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        raise ValueError("No PDF documents found in the specified directory.")

    print(f"✅ Loaded {len(documents)} documents.")
    return documents


def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split the documents into smaller text chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = splitter.split_documents(documents)
    print(f"✅ Created {len(texts)} text chunks.")
    return texts


def create_vector_db():
    """Create a FAISS vector database from PDF documents."""
    try:
        print("📄 Loading documents...")
        documents = load_documents(DATA_PATH)

        print("✂️ Splitting documents into chunks...")
        texts = split_documents(documents)

        print("🧠 Creating embeddings (this may take a while)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        print("💾 Building FAISS vector store...")
        db = FAISS.from_documents(texts, embeddings)

        os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
        db.save_local(DB_FAISS_PATH)

        print(f"✅ Vector database saved to: {DB_FAISS_PATH}")
        print(f"✅ Total chunks indexed: {len(texts)}")

        # Optional: quick search test
        test_search(db)

    except Exception as e:
        print(f"❌ An error occurred while creating FAISS DB: {e}")
        import traceback
        traceback.print_exc()


def test_search(db):
    """Quick test to verify FAISS search is working."""
    print("\n--- 🔍 Testing FAISS search ---")
    queries = [
        "What are the symptoms of diabetes?",
        "How to treat high blood pressure?",
        "What is cardiovascular disease?",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        results = db.similarity_search(q, k=1)
        if results:
            print(f"Result: {results[0].page_content[:200]}...\n")
        else:
            print("No relevant results found.\n")


if __name__ == "__main__":
    print("=" * 70)
    print(" 🧩 Medical Chatbot - FAISS Database Creation")
    print("=" * 70)
    create_vector_db()
    print("\n✅ Process completed successfully!")
