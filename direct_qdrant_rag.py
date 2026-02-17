"""
Budget RAG with Direct Qdrant Access
=====================================
Bypasses LangChain to avoid version conflicts
"""

import sys
import os
sys.path.append(os.path.abspath(".."))

from pathlib import Path
from typing import List, TypedDict, Optional, Dict
from contextlib import asynccontextmanager
from datetime import datetime
import uuid

# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Direct Qdrant access
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# LangChain Documents
from langchain_core.documents import Document

# Groq
from langchain_groq import ChatGroq

# LangGraph
from langgraph.graph import StateGraph, END

# ============================================
# CONFIGURATION
# ============================================

FOLDER_PATH = "C:/Users/ashwi/Documents/Budgets"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "indian_budget"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ⚠️ PUT YOUR GROQ API KEY HERE
GROQ_API_KEY = ""

# ============================================
# GLOBAL VARIABLES
# ============================================

qdrant_client = None
embeddings = None
llm = None
rag_app = None
chat_sessions: Dict[str, List[dict]] = {}

# ============================================
# PYDANTIC MODELS
# ============================================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class SourceInfo(BaseModel):
    source: str
    page: int

class ChatResponse(BaseModel):
    response: str
    sources: List[SourceInfo]
    session_id: str

# ============================================
# DIRECT QDRANT RETRIEVAL
# ============================================

def search_qdrant(query: str, k: int = 5) -> List[Document]:
    """
    Search Qdrant directly without LangChain wrapper
    """
    # Convert query to vector
    query_vector = embeddings.embed_query(query)
    
    # Search using Qdrant client directly
    # Try new API first, fall back to old API if needed
    try:
        # New API (qdrant-client >= 1.7)
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=k,
            with_payload=True
        ).points
    except AttributeError:
        # Old API (qdrant-client < 1.7)
        from qdrant_client.http import models as qmodels
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=k,
            with_payload=True
        )
    
    # Convert to LangChain Documents
    documents = []
    for result in results:
        # Extract payload
        payload = result.payload if hasattr(result, 'payload') else {}
        
        # Get text content
        page_content = payload.get('page_content', '')
        
        # Get metadata
        metadata = payload.get('metadata', {})
        
        # Create Document
        doc = Document(
            page_content=page_content,
            metadata=metadata
        )
        documents.append(doc)
    
    return documents

# ============================================
# INITIALIZATION
# ============================================

def initialize_qdrant():
    """Connect to Qdrant"""
    global qdrant_client, embeddings
    
    print("💾 Connecting to Qdrant...")
    
    try:
        # Create client
        qdrant_client = QdrantClient(url=QDRANT_URL)
        
        # Check collection exists
        collections = qdrant_client.get_collections().collections
        if not any(c.name == COLLECTION_NAME for c in collections):
            raise Exception(f"Collection '{COLLECTION_NAME}' not found!")
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        info = qdrant_client.get_collection(COLLECTION_NAME)
        print(f"✅ Qdrant connected ({info.points_count} vectors)")
        
    except Exception as e:
        raise Exception(f"Qdrant error: {e}")

def initialize_llm():
    """Initialize Groq"""
    global llm
    
    print("\n🤖 Initializing Groq...")
    
    if GROQ_API_KEY == "your-groq-api-key-here":
        raise Exception("❌ Set GROQ_API_KEY in code!")
    
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            api_key=GROQ_API_KEY,
            max_tokens=512
        )
        
        llm.invoke("test")
        print("✅ Groq ready")
        
    except Exception as e:
        raise Exception(f"Groq error: {e}")

def build_rag_graph():
    """Build RAG"""
    global rag_app
    
    class RAGState(TypedDict):
        question: str
        retrieved_docs: List[Document]
        context: str
        answer: str
    
    def retrieve_documents(state: RAGState):
        """Retrieve using direct Qdrant access"""
        question = state['question'].strip()
        print(f"\n🔍 Question: {question}")
        
        try:
            # ✅ DIRECT QDRANT SEARCH - Works with all versions
            docs = search_qdrant(question, k=5)
            
            print(f"   Retrieved {len(docs)} documents")
            for i, doc in enumerate(docs, 1):
                src = doc.metadata.get('source', 'Unknown')
                pg = doc.metadata.get('page', 'N/A')
                print(f"      [{i}] {src} - Page {pg}")
            
            return {"retrieved_docs": docs}
            
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
            import traceback
            traceback.print_exc()
            return {"retrieved_docs": []}
    
    def format_context(state: RAGState):
        """Format context"""
        if not state["retrieved_docs"]:
            return {"context": ""}
        
        formatted = []
        for i, doc in enumerate(state["retrieved_docs"], 1):
            src = doc.metadata.get('source', 'Unknown')
            pg = doc.metadata.get('page', 'N/A')
            txt = " ".join(doc.page_content.split())
            formatted.append(f"[Doc {i}: {src}, Page {pg}]\n{txt}")
        
        return {"context": "\n\n".join(formatted)}
    
    def generate_answer(state: RAGState):
        """Generate answer"""
        if not state["context"]:
            return {"answer": "No relevant information found in the budget documents."}
        
        prompt = f"""Answer based on these Indian Budget documents:

{state['context'][:3000]}

Question: {state['question']}

Provide a clear, specific answer (2-4 sentences):"""
        
        try:
            response = llm.invoke(prompt)
            answer = response.content.strip()
            print(f"   ✅ Generated answer ({len(answer)} chars)")
            return {"answer": answer}
            
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            
            try:
                # Fallback model
                fallback = ChatGroq(
                    model="llama-3.1-8b-instant",
                    api_key=GROQ_API_KEY,
                    temperature=0.3
                )
                response = fallback.invoke(prompt)
                return {"answer": response.content.strip()}
            except:
                return {"answer": "Error generating answer. Please try again."}
    
    # Build graph
    print("\n🔧 Building RAG...")
    workflow = StateGraph(RAGState)
    
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("format", format_context)
    workflow.add_node("generate", generate_answer)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "format")
    workflow.add_edge("format", "generate")
    workflow.add_edge("generate", END)
    
    rag_app = workflow.compile()
    print("✅ RAG ready!")

# ============================================
# FASTAPI
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*70)
    print("🚀 DIRECT QDRANT ACCESS - GROQ RAG")
    print("="*70)
    
    try:
        initialize_qdrant()
        initialize_llm()
        build_rag_graph()
        
        print("\n" + "="*70)
        print("✅ READY")
        print("="*70)
        print("\n📍 http://localhost:8000")
        print("📚 http://localhost:8000/docs\n")
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}\n")
        raise
    
    yield

app = FastAPI(title="Direct Qdrant Groq RAG", version="6.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "online", "version": "6.0.0"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message:
        raise HTTPException(400, "Empty message")
    
    session_id = request.session_id or str(uuid.uuid4())
    
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    try:
        result = rag_app.invoke({
            "question": request.message,
            "retrieved_docs": [],
            "context": "",
            "answer": ""
        })
        
        sources = [
            SourceInfo(
                source=doc.metadata.get('source', 'Unknown'),
                page=doc.metadata.get('page', 0)
            )
            for doc in result.get('retrieved_docs', [])
        ]
        
        chat_sessions[session_id].append({
            "question": request.message,
            "answer": result['answer'],
            "sources": [{"source": s.source, "page": s.page} for s in sources],
            "timestamp": datetime.now().isoformat()
        })
        
        return ChatResponse(
            response=result['answer'],
            sources=sources,
            session_id=session_id
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(500, str(e))

@app.get("/sessions")
async def get_sessions():
    sessions = []
    for sid, msgs in chat_sessions.items():
        if msgs:
            sessions.append({
                "session_id": sid,
                "message_count": len(msgs),
                "first_message": msgs[0]["question"][:50],
                "last_updated": msgs[-1]["timestamp"]
            })
    return {"sessions": sorted(sessions, key=lambda x: x["last_updated"], reverse=True)}

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    if session_id not in chat_sessions:
        raise HTTPException(404, "Not found")
    return {"messages": chat_sessions[session_id]}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {"status": "deleted"}
    raise HTTPException(404, "Not found")

@app.get("/health")
async def health():
    return {
        "qdrant": qdrant_client is not None,
        "llm": llm is not None,
        "rag_app": rag_app is not None,
        "status": "healthy" if all([qdrant_client, llm, rag_app]) else "unhealthy"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
