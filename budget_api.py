import sys
import os
sys.path.append(os.path.abspath(".."))

from pathlib import Path
from typing import List, TypedDict, Optional
from contextlib import asynccontextmanager

# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Document processing
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Embeddings & Vector Store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams

# LLM
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# LangGraph
from langgraph.graph import StateGraph, END

# ============================================
# CONFIGURATION
# ============================================

FOLDER_PATH = "C:/Users/ashwi/Documents/indian-budget-rag/data/raw/"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "indian_budget"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

LLM_MODEL = "google/flan-t5-base"                      # Original (fast but lower quality)
# LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"         # Best for laptops
# LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"        # Best quality
# LLM_MODEL = "meta-llama/Llama-3.2-3B-Instruct"         # Good balance
# LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"             # Best chat

# ============================================
# GLOBAL VARIABLES (will be initialized on startup)
# ============================================

vectorstore = None
llm = None
rag_app = None

# ============================================
# PYDANTIC MODELS
# ============================================

class ChatRequest(BaseModel):
    message: str

class SourceInfo(BaseModel):
    source: str
    page: int

class ChatResponse(BaseModel):
    response: str
    sources: List[SourceInfo]

# ============================================
# HELPER FUNCTIONS
# ============================================

def load_budget_pdfs_with_metadata(folder_path: str) -> List[Document]:
    """Load all PDFs from folder with metadata"""
    all_docs = []
    folder_path = Path(folder_path)
    pdf_files = list(folder_path.glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDFs:")
    for f in pdf_files:
        print(f"  - {f.name}")
    
    for pdf_file in pdf_files:
        loader = PyMuPDFLoader(str(pdf_file))
        docs = loader.load()
        
        for i, doc in enumerate(docs):
            doc.metadata["source"] = pdf_file.name
            doc.metadata["page"] = i + 1
        
        all_docs.extend(docs)
    
    print(f"Loaded {len(all_docs)} pages from {len(pdf_files)} PDFs")
    
    return all_docs

def initialize_vectorstore():
    """Initialize or load vectorstore"""
    global vectorstore
    
    print("📄 Loading PDFs...")
    docs = load_budget_pdfs_with_metadata(FOLDER_PATH)
    
    print("\n✂️ Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    
    print("\n🔢 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    
    print("\n💾 Setting up Qdrant...")
    client = QdrantClient(url=QDRANT_URL)
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_exists = any(c.name == COLLECTION_NAME for c in collections)
    
    if not collection_exists:
        print(f"  Creating new collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance="Cosine"
            )
        )
        
        # Create vectorstore and add documents
        vectorstore = Qdrant(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings
        )
        
        print("\n📥 Adding documents to vectorstore...")
        vectorstore.add_documents(chunks)
        print("✅ Vectorstore created and populated!")
    else:
        print(f"  Using existing collection: {COLLECTION_NAME}")
        vectorstore = Qdrant(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings
        )
        print("✅ Vectorstore loaded!")

def initialize_llm():
    """Initialize the language model"""
    global llm
    
    print("\n🤖 Loading language model...")
    hf_pipeline = pipeline(
        "text2text-generation",
        model=LLM_MODEL,
        max_length=512,
        device=-1  # CPU
    )
    
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    print("✅ LLM ready!")

def build_rag_graph():
    """Build the LangGraph RAG workflow"""
    global rag_app
    
    class RAGState(TypedDict):
        question: str
        retrieved_docs: List[Document]
        context: str
        answer: str
    
    def retrieve_documents(state: RAGState):
        """Node 1: Retrieve relevant documents"""
        print(f"\n🔍 Retrieving docs for: {state['question']}")
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(state["question"])
        
        print(f"   Found {len(docs)} documents")
        
        return {"retrieved_docs": docs}
    
    def format_context(state: RAGState):
        """Node 2: Format documents into context"""
        print("📝 Formatting context...")
        
        formatted = []
        for i, doc in enumerate(state["retrieved_docs"], 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            content = doc.page_content.strip()
            formatted.append(f"[Doc {i} - {source}, Page {page}]\n{content}")
        
        context = "\n\n".join(formatted)
        
        return {"context": context}
    
    def generate_answer(state: RAGState):
        """Node 3: Generate answer using LLM"""
        print("🤖 Generating answer with LLM...")
        
        prompt = f"""Based on the following budget documents, answer the question concisely.

Context from Indian Budget Documents:
{state['context']}

Question: {state['question']}

Answer:"""
        
        # Generate response
        response = llm.invoke(prompt)
        
        # Extract text from response
        if isinstance(response, str):
            answer = response
        else:
            answer = str(response)
        
        answer = answer.strip()
        
        print(f"   Generated {len(answer)} characters")
        
        return {"answer": answer}
    
    # Build the graph
    print("\n🔧 Building LangGraph workflow...")
    workflow = StateGraph(RAGState)
    
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("format", format_context)
    workflow.add_node("generate", generate_answer)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "format")
    workflow.add_edge("format", "generate")
    workflow.add_edge("generate", END)
    
    rag_app = workflow.compile()
    print("✅ Graph compiled!")

# ============================================
# FASTAPI LIFESPAN MANAGEMENT
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, cleanup on shutdown"""
    print("\n" + "="*70)
    print("🚀 STARTING INDIAN BUDGET RAG API")
    print("="*70)
    
    # Startup: Initialize all components
    try:
        initialize_vectorstore()
        initialize_llm()
        build_rag_graph()
        
        print("\n" + "="*70)
        print("✅ API READY TO SERVE REQUESTS")
        print("="*70 + "\n")
    except Exception as e:
        print(f"\n❌ Error during startup: {e}")
        raise
    
    yield
    
    # Shutdown: Cleanup if needed
    print("\n👋 Shutting down...")

# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="Indian Budget RAG API",
    description="Chatbot API for querying Indian Budget documents using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Indian Budget RAG API",
        "version": "1.0.0"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for the Indian Budget chatbot
    
    Args:
        request: ChatRequest containing the user's message
    
    Returns:
        ChatResponse with the answer and source documents
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        print(f"\n📨 Received question: {request.message}")
        
        # Run the RAG pipeline
        result = rag_app.invoke({
            "question": request.message,
            "retrieved_docs": [],
            "context": "",
            "answer": ""
        })
        
        # Extract sources
        sources = [
            SourceInfo(
                source=doc.metadata.get('source', 'Unknown'),
                page=doc.metadata.get('page', 0)
            )
            for doc in result['retrieved_docs']
        ]
        
        # Create response
        response = ChatResponse(
            response=result['answer'],
            sources=sources
        )
        
        print(f"✅ Response sent: {len(result['answer'])} characters")
        
        return response
        
    except Exception as e:
        print(f"❌ Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    """Check if all components are initialized"""
    return {
        "vectorstore": vectorstore is not None,
        "llm": llm is not None,
        "rag_app": rag_app is not None,
        "status": "healthy" if all([vectorstore, llm, rag_app]) else "initializing"
    }

# ============================================
# RUN THE SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("Starting FastAPI server...")
    print("="*70)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
