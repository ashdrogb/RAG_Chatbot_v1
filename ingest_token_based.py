"""
Token-Based PDF Ingestion Script
=================================
Re-chunks all PDFs using token-based splitting (300 tokens, 50 overlap)
Deletes old collection and creates a fresh one

WARNING: This will DELETE your existing collection!
"""

from pathlib import Path
from typing import List
import tiktoken

# Document processing
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Embeddings & Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# ============================================
# CONFIGURATION
# ============================================

FOLDER_PATH = "C:/Users/ashwi/Documents/Budgets"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "indian_budget"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Token-based chunking settings
CHUNK_SIZE = 300        # tokens
CHUNK_OVERLAP = 50      # tokens

# ============================================
# TOKEN COUNTER
# ============================================

def get_token_counter():
    """
    Returns a function that counts tokens using tiktoken
    Uses cl100k_base encoding (GPT-4/ChatGPT encoding)
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(text: str) -> int:
        return len(encoding.encode(text))
    
    return count_tokens

# ============================================
# LOAD PDFs
# ============================================

def load_budget_pdfs(folder_path: str) -> List[Document]:
    """Load all PDFs with metadata"""
    print(f"\n📄 Loading PDFs from: {folder_path}")
    
    all_docs = []
    folder_path = Path(folder_path)
    pdf_files = list(folder_path.glob("*.pdf"))
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in {folder_path}")
    
    print(f"   Found {len(pdf_files)} PDFs:")
    for pdf_file in pdf_files:
        print(f"      - {pdf_file.name}")
    
    print("\n   Loading content...")
    for pdf_file in pdf_files:
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            docs = loader.load()
            
            # Add metadata
            for i, doc in enumerate(docs):
                doc.metadata["source"] = pdf_file.name
                doc.metadata["page"] = i + 1
            
            all_docs.extend(docs)
            print(f"      ✅ {pdf_file.name}: {len(docs)} pages")
            
        except Exception as e:
            print(f"      ❌ {pdf_file.name}: Error - {e}")
    
    print(f"\n   ✅ Total: {len(all_docs)} pages loaded")
    return all_docs

# ============================================
# TOKEN-BASED CHUNKING
# ============================================

def create_token_based_chunks(documents: List[Document]) -> List[Document]:
    """
    Split documents using token-based chunking
    """
    print(f"\n✂️  Creating token-based chunks...")
    print(f"   Settings:")
    print(f"      Chunk size: {CHUNK_SIZE} tokens")
    print(f"      Overlap: {CHUNK_OVERLAP} tokens")
    
    # Get token counter
    token_counter = get_token_counter()
    
    # Create splitter with token-based length function
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=token_counter,  # ✅ Token-based!
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True
    )
    
    # Split documents
    chunks = text_splitter.split_documents(documents)
    
    # Add chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_tokens"] = token_counter(chunk.page_content)
    
    print(f"\n   ✅ Created {len(chunks)} chunks")
    
    # Show statistics
    token_counts = [chunk.metadata["chunk_tokens"] for chunk in chunks]
    avg_tokens = sum(token_counts) / len(token_counts)
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    
    print(f"\n   📊 Statistics:")
    print(f"      Average tokens per chunk: {avg_tokens:.1f}")
    print(f"      Min tokens: {min_tokens}")
    print(f"      Max tokens: {max_tokens}")
    
    return chunks

# ============================================
# QDRANT SETUP
# ============================================

def setup_qdrant_collection(client: QdrantClient):
    """
    Delete old collection and create new one
    """
    print(f"\n💾 Setting up Qdrant collection...")
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if COLLECTION_NAME in collection_names:
        print(f"   ⚠️  Collection '{COLLECTION_NAME}' exists")
        response = input("   Delete and recreate? (yes/no): ")
        
        if response.lower() != 'yes':
            print("   ❌ Aborted. No changes made.")
            return False
        
        print(f"   🗑️  Deleting old collection...")
        client.delete_collection(COLLECTION_NAME)
        print(f"   ✅ Deleted")
    
    # Create new collection
    print(f"   📦 Creating new collection...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE
        )
    )
    print(f"   ✅ Collection created")
    
    return True

# ============================================
# EMBED AND UPLOAD
# ============================================

def upload_to_qdrant(chunks: List[Document], client: QdrantClient):
    """
    Create embeddings and upload to Qdrant
    """
    print(f"\n🔢 Creating embeddings and uploading...")
    
    # Initialize embeddings
    print(f"   Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Process in batches
    batch_size = 100
    total_chunks = len(chunks)
    
    print(f"\n   Uploading {total_chunks} chunks in batches of {batch_size}...")
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        print(f"      Batch {batch_num}/{total_batches} ({len(batch)} chunks)...", end=" ")
        
        try:
            # Get texts and metadatas
            texts = [chunk.page_content for chunk in batch]
            metadatas = [chunk.metadata for chunk in batch]
            
            # Create embeddings
            vectors = embeddings.embed_documents(texts)
            
            # Upload to Qdrant
            from qdrant_client.models import PointStruct
            
            points = [
                PointStruct(
                    id=i + j,
                    vector=vector,
                    payload={
                        "page_content": text,
                        "metadata": metadata
                    }
                )
                for j, (vector, text, metadata) in enumerate(zip(vectors, texts, metadatas))
            ]
            
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            
            print("✅")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
    
    print(f"\n   ✅ All chunks uploaded!")

# ============================================
# MAIN INGESTION PIPELINE
# ============================================

def main():
    """
    Main ingestion pipeline
    """
    print("="*70)
    print("🚀 TOKEN-BASED PDF INGESTION")
    print("="*70)
    print(f"\nThis will:")
    print(f"1. Load all PDFs from: {FOLDER_PATH}")
    print(f"2. Chunk using token-based splitting ({CHUNK_SIZE} tokens)")
    print(f"3. Delete existing '{COLLECTION_NAME}' collection")
    print(f"4. Create new collection and upload chunks")
    print("\n⚠️  WARNING: This will DELETE your existing collection!")
    print("="*70)
    
    proceed = input("\nProceed? (yes/no): ")
    if proceed.lower() != 'yes':
        print("\n❌ Aborted. No changes made.")
        return
    
    try:
        # Step 1: Load PDFs
        documents = load_budget_pdfs(FOLDER_PATH)
        
        # Step 2: Create token-based chunks
        chunks = create_token_based_chunks(documents)
        
        # Step 3: Connect to Qdrant
        print(f"\n🔌 Connecting to Qdrant at {QDRANT_URL}...")
        client = QdrantClient(url=QDRANT_URL)
        print(f"   ✅ Connected")
        
        # Step 4: Setup collection
        if not setup_qdrant_collection(client):
            return
        
        # Step 5: Upload chunks
        upload_to_qdrant(chunks, client)
        
        # Step 6: Verify
        print(f"\n✅ INGESTION COMPLETE!")
        print("="*70)
        
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"\n📊 Final Statistics:")
        print(f"   Collection: {COLLECTION_NAME}")
        print(f"   Total vectors: {collection_info.points_count}")
        print(f"   Vector dimension: {EMBEDDING_DIM}")
        print(f"   Chunking method: Token-based")
        print(f"   Chunk size: {CHUNK_SIZE} tokens")
        print(f"   Overlap: {CHUNK_OVERLAP} tokens")
        
        print("\n✅ You can now restart your RAG server!")
        print("   python direct_qdrant_rag.py")
        
    except Exception as e:
        print(f"\n❌ INGESTION FAILED: {e}")
        import traceback
        traceback.print_exc()

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    main()
