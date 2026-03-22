# Enterprise Legal Contract Analyzer (RAG System)

A production-ready, local-first Retrieval-Augmented Generation (RAG) system designed for analyzing legal contracts. Upload contracts, ask natural language questions, and receive AI-powered answers with full source citations.

---

## Overview

This system enables legal professionals to:

- **Upload** legal contracts (PDF, TXT, DOC, DOCX)
- **Ask** natural language questions about clauses, terms, and obligations
- **Receive** accurate, context-grounded answers with traceable citations
- **Filter** searches to specific documents or search across the entire corpus

Built with a local-first architecture, all document processing and embedding generation happen on your GPU—no data leaves your machine.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (Next.js)                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Upload    │  │   Chat UI   │  │  Metadata   │  │  Citation Accordion │ │
│  │   Sidebar   │  │  Interface  │  │   Filter    │  │    (Sources UI)     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ HTTP (localhost:3000 → :8000)
┌───────────────────────────────▼─────────────────────────────────────────────┐
│                              BACKEND (FastAPI)                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   /ingest   │  │    /ask     │  │   Intent    │  │   Prompt Builder    │ │
│  │   Endpoint  │  │   Endpoint  │  │   Router    │  │   (Hybrid Agent)    │ │
│  └──────┬──────┘  └──────┬──────┘  └─────────────┘  └─────────────────────┘ │
│         │                │                                                   │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌─────────────────────────────────────┐  │
│  │ Unstructured│  │  Embedding  │  │           LlamaIndex Core           │  │
│  │  (Parsing)  │  │   (GPU)     │  │         (Orchestration)             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
┌─────────▼─────────┐ ┌─────────▼─────────┐ ┌─────────▼─────────┐
│   Endee Vector    │ │  HuggingFace BGE  │ │    Groq Cloud     │
│   Database        │ │  (Local GPU)      │ │    Llama-3 LLM    │
│   (Docker)        │ │  1024-dim vectors │ │    (Inference)    │
└───────────────────┘ └───────────────────┘ └───────────────────┘
```

---

## Technology Stack

### Frontend
| Technology | Purpose |
|------------|---------|
| **Next.js 15** | React framework with App Router |
| **React 19** | UI components |
| **Tailwind CSS 4** | Utility-first styling |
| **TypeScript** | Type safety |

### Backend
| Technology | Purpose |
|------------|---------|
| **FastAPI** | High-performance async API |
| **LlamaIndex** | RAG orchestration framework |
| **Endee** | Local vector database (Docker) |
| **HuggingFace** | BAAI/bge-large-en-v1.5 embeddings (1024-dim) |
| **Groq** | Llama-3.1-8B-Instant inference |
| **Unstructured** | Document parsing (PDF, DOCX, TXT) |

---

## Key Features

### Intent Routing
Greetings and general questions bypass the vector database, reducing latency and preventing unnecessary API calls.

### Metadata Filtering
Search within a specific uploaded document or across the entire contract corpus using the dropdown filter.

### Local-First GPU Processing
All embeddings are generated locally on your NVIDIA GPU using the BGE-Large model—no data sent to external embedding services.

### Citation UI
Every AI response includes a collapsible "Sources & Citations" accordion showing the exact contract chunks used to generate the answer.

### Hybrid Agent Prompt
The LLM operates as a dual-persona legal assistant: a friendly conversationalist for general queries, and a precise legal analyst for contract questions.

---

## Local Setup

### Prerequisites

- **Node.js** 18+ (for frontend)
- **Python** 3.10+ (for backend)
- **Docker** (for Endee vector database)
- **NVIDIA GPU** with CUDA support (for local embeddings)
- **Groq API Key** (free tier available at [console.groq.com](https://console.groq.com))

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/contract-analyzer.git
cd contract-analyzer
```

### 2. Start the Vector Database

```bash
docker run -d --name endee -p 8080:8080 endeeai/endee:latest
```

### 3. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# Start the server
python main.py
```

The backend will be available at `http://localhost:8000`.

### 4. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:3000`.

---

## Usage

1. **Upload a Contract**: Use the sidebar to upload a PDF, TXT, DOC, or DOCX file
2. **Wait for Processing**: The document is chunked, embedded, and stored in the vector database
3. **Ask Questions**: Type natural language questions in the chat interface
4. **View Citations**: Expand the "Sources & Citations" accordion to see the exact text used

### Example Questions

- "What are the termination clauses in this agreement?"
- "Summarize the payment terms and conditions"
- "What are the confidentiality obligations for both parties?"
- "Are there any non-compete restrictions?"

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest` | Upload and process a document |
| `POST` | `/ask` | Submit a question and receive an answer |
| `GET` | `/health` | Health check endpoint |

### POST /ask Request Body

```json
{
  "query": "What are the payment terms?",
  "file_name": "contract.pdf"  // Optional: filter to specific file
}
```

---

## Project Structure

```
contract-analyzer/
├── frontend/
│   ├── app/
│   │   ├── page.tsx        # Main chat interface
│   │   ├── layout.tsx      # Root layout
│   │   └── globals.css     # Tailwind styles
│   ├── package.json
│   └── ...
├── backend/
│   ├── main.py             # FastAPI application
│   ├── batch_ingest.py     # CUAD dataset bulk loader
│   ├── requirements.txt
│   └── ...
├── .gitignore
└── README.md
```

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/) - RAG framework
- [Groq](https://groq.com/) - Ultra-fast LLM inference
- [Endee](https://endee.ai/) - Local vector database
- [CUAD Dataset](https://www.atticusprojectai.org/cuad) - Contract Understanding Atticus Dataset
