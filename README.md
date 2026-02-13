# Indian Budget RAG Chatbot

A conversational AI chatbot that answers questions about Indian Budget documents using Retrieval-Augmented Generation (RAG).

## Architecture

- **Backend**: FastAPI server with LangGraph RAG pipeline
- **Vector Store**: Qdrant for document embeddings
- **LLM**: Google FLAN-T5 (local)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Frontend**: Simple HTML/CSS/JavaScript chatbot UI

## Prerequisites

1. **Python 3.8+**
2. **Qdrant** (running on `localhost:6333`)
3. **PDF documents** in the configured folder

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Qdrant

If you haven't already, start Qdrant using Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Or install locally: https://qdrant.tech/documentation/quick-start/

### 3. Configure Paths

Edit `budget_api.py` and update the `FOLDER_PATH` to point to your PDF directory:

```python
FOLDER_PATH = "C:/Users/ashwi/Documents/indian-budget-rag/data/raw/"
```

## Usage

### 1. Start the FastAPI Server

```bash
python budget_api.py
```

Or using uvicorn directly:

```bash
uvicorn budget_api:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
🚀 STARTING INDIAN BUDGET RAG API
📄 Loading PDFs...
✂️ Splitting documents...
🔢 Creating embeddings...
💾 Setting up Qdrant...
🤖 Loading language model...
✅ API READY TO SERVE REQUESTS
```

### 2. Open the Chatbot UI

Simply open `indian_budget_chatbot.html` in your browser. The chatbot will automatically connect to `http://localhost:8000`.

### 3. Start Asking Questions!

Example questions:
- "What were the tax changes in Budget 2020-21?"
- "What is the fiscal deficit mentioned?"
- "What allocations were made for education?"
- "Tell me about infrastructure spending"

## API Endpoints

### `POST /chat`

Send a message to the chatbot.

**Request:**
```json
{
  "message": "What were the tax changes?"
}
```

**Response:**
```json
{
  "response": "The budget introduced several tax changes...",
  "sources": [
    {
      "source": "budget_2020.pdf",
      "page": 15
    },
    {
      "source": "budget_2021.pdf",
      "page": 8
    }
  ]
}
```

### `GET /health`

Check if all components are initialized.

**Response:**
```json
{
  "vectorstore": true,
  "llm": true,
  "rag_app": true,
  "status": "healthy"
}
```

## Project Structure

```
.
├── budget_api.py              # FastAPI server with RAG pipeline
├── indian_budget_chatbot.html # Frontend UI
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Configuration Options

You can modify these constants in `budget_api.py`:

```python
FOLDER_PATH = "path/to/pdfs"              # Where your PDFs are stored
QDRANT_URL = "http://localhost:6333"     # Qdrant server URL
COLLECTION_NAME = "indian_budget"         # Qdrant collection name
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"        # Can use flan-t5-large for better quality
```

## Performance Tips

1. **First run is slow**: The first time you start the server, it needs to:
   - Load all PDFs
   - Create embeddings for all chunks
   - Load the LLM model
   - This can take 5-10 minutes depending on your hardware

2. **Subsequent runs are fast**: Once the Qdrant collection is created, it's reused on subsequent runs

3. **Upgrade the LLM**: For better answers, use a larger model:
   ```python
   LLM_MODEL = "google/flan-t5-large"  # or flan-t5-xl
   ```

4. **GPU acceleration**: If you have a GPU, change:
   ```python
   device=-1  # CPU
   ```
   to:
   ```python
   device=0   # GPU
   ```

## Troubleshooting

### "Connection refused" error in the UI

- Make sure the FastAPI server is running on port 8000
- Check the browser console for CORS errors
- Verify the API endpoint in the HTML file matches your server

### Qdrant connection errors

- Ensure Qdrant is running: `docker ps` should show the qdrant container
- Check that port 6333 is not blocked by a firewall

### Out of memory errors

- Reduce `chunk_size` in the text splitter
- Use a smaller LLM model (flan-t5-small)
- Process fewer PDFs at once

### Slow responses

- The first query is always slower (model warmup)
- Consider using a GPU for faster inference
- Reduce the number of retrieved documents (change `k` value)

## API Testing with curl

Test the API directly:

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the fiscal deficit?"}'
```

## License

MIT
