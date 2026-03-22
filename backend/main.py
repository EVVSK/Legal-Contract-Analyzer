"""
Enterprise Legal Contract Analyzer - FastAPI Backend
=====================================================
This module initializes the FastAPI application and configures
LlamaIndex with our local-first RAG architecture.

Stack:
- LLM: GroqCloud API (llama3-8b-8192)
- Embeddings: Local HuggingFace (BAAI/bge-large-en-v1.5) on GPU
- Vector DB: Endee (local Docker instance)
"""

import os
import logging
import tempfile
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from unstructured.partition.auto import partition

from endee import Endee

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMENSION = 1024  # BGE-large-en-v1.5 produces 1024-dim vectors
LLM_MODEL_NAME = "llama-3.1-8b-instant"
FRONTEND_ORIGIN = "http://localhost:3000"

# Endee Vector Database Configuration
ENDEE_BASE_URL = "http://localhost:8080/api/v1"
ENDEE_INDEX_NAME = "contracts"

# Chunking Configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# RAG Configuration
RAG_TOP_K = 12  # Number of similar chunks to retrieve (balanced for token limits)
RAG_TOP_K_FILTERED = 50  # Fetch more results when filtering to ensure enough matches
CONTEXT_MAX_CHARS = 20000  # Maximum context length to prevent Groq rate limits

# Intent Routing - Greeting patterns to bypass database search
GREETING_PATTERNS = [
    "hi", "hello", "hey", "howdy", "greetings",
    "what can you do", "who are you", "what are you",
    "help", "help me", "how do you work", "how does this work",
    "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "bye", "goodbye"
]

# -----------------------------------------------------------------------------
# Initialize Endee Client Globally
# -----------------------------------------------------------------------------
endee_client = Endee(token="local-dev-token")
endee_client.set_base_url(ENDEE_BASE_URL)

logger.info(f"Endee client initialized with base URL: {ENDEE_BASE_URL}")


# -----------------------------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------------------------
class QueryRequest(BaseModel):
    """Request model for the /ask endpoint."""
    query: str
    file_name: str | None = None  # Optional: filter search to specific file


# -----------------------------------------------------------------------------
# Lifespan Context Manager (Modern FastAPI Startup/Shutdown)
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.

    On startup:
        - Loads environment variables from .env
        - Initializes the local HuggingFace embedding model on GPU
        - Configures the Groq LLM client with API key
        - Sets LlamaIndex global settings

    On shutdown:
        - Performs cleanup (if needed in future)
    """
    logger.info("=" * 60)
    logger.info("Starting Enterprise Legal Contract Analyzer...")
    logger.info("=" * 60)

    # Load environment variables from .env file
    load_dotenv()

    # Retrieve and validate GROQ API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("GROQ_API_KEY not found in environment variables!")
        raise ValueError("GROQ_API_KEY is required. Please set it in the .env file.")

    logger.info("GROQ_API_KEY loaded successfully.")

    # -------------------------------------------------------------------------
    # Initialize Embedding Model (Local HuggingFace on GPU)
    # -------------------------------------------------------------------------
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    logger.info("This will run locally on your GPU (GTX 1650 4GB VRAM)...")

    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        device="cuda",  # Use GPU for inference
        trust_remote_code=True
    )

    logger.info("Embedding model loaded successfully on GPU.")

    # -------------------------------------------------------------------------
    # Initialize LLM (GroqCloud API)
    # -------------------------------------------------------------------------
    logger.info(f"Configuring LLM: {LLM_MODEL_NAME} via GroqCloud")

    llm = Groq(
        model=LLM_MODEL_NAME,
        api_key=groq_api_key
    )

    logger.info("Groq LLM client configured successfully.")

    # -------------------------------------------------------------------------
    # Configure LlamaIndex Global Settings
    # -------------------------------------------------------------------------
    Settings.embed_model = embed_model
    Settings.llm = llm

    logger.info("LlamaIndex Settings configured.")
    logger.info("=" * 60)
    logger.info("System initialization complete. Server is ready.")
    logger.info("=" * 60)

    # Yield control to the application
    yield

    # Shutdown logic
    logger.info("Shutting down Enterprise Legal Contract Analyzer...")
    logger.info("Cleanup complete. Goodbye.")


# -----------------------------------------------------------------------------
# FastAPI Application Instance
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Enterprise Legal Contract Analyzer",
    description="A local-first RAG system for analyzing legal contracts using LlamaIndex, Groq, and Endee.",
    version="1.0.0",
    lifespan=lifespan
)

# -----------------------------------------------------------------------------
# CORS Middleware Configuration
# -----------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],  # Next.js frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

logger.info(f"CORS enabled for origin: {FRONTEND_ORIGIN}")


# -----------------------------------------------------------------------------
# Health Check Endpoint
# -----------------------------------------------------------------------------
@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint to verify system status.

    Returns:
        dict: System status including vector DB and LLM information.
    """
    return {
        "status": "System Online",
        "vector_db": "Endee",
        "llm": "Groq Llama-3"
    }


# -----------------------------------------------------------------------------
# Document Ingestion Endpoint
# -----------------------------------------------------------------------------
@app.post("/ingest", tags=["Ingestion"])
async def ingest_document(file: UploadFile = File(...)):
    """
    Ingests a document (PDF, DOCX, TXT, etc.) into the vector database.

    Process:
        1. Save uploaded file to temporary location
        2. Extract text using unstructured library
        3. Split text into chunks using LlamaIndex SentenceSplitter
        4. Generate embeddings for each chunk using local HuggingFace model
        5. Upsert vectors with metadata to Endee vector database
        6. Clean up temporary file

    Args:
        file: The uploaded document file

    Returns:
        dict: Summary of ingestion results including chunk count

    Raises:
        HTTPException: If text extraction, embedding, or database upsert fails
    """
    temp_file_path = None

    try:
        # ---------------------------------------------------------------------
        # Step 1: Save uploaded file to temporary location
        # ---------------------------------------------------------------------
        logger.info(f"Received file for ingestion: {file.filename}")

        # Preserve the original file extension for unstructured to detect type
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ""

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        logger.info(f"File saved to temporary location: {temp_file_path}")

        # ---------------------------------------------------------------------
        # Step 2: Extract text using unstructured
        # ---------------------------------------------------------------------
        logger.info("Extracting text from document using unstructured...")

        elements = partition(filename=temp_file_path)

        if not elements:
            raise HTTPException(
                status_code=400,
                detail="Could not extract any text from the uploaded file."
            )

        # Combine all extracted elements into a single text string
        raw_text = "\n\n".join([str(element) for element in elements])
        logger.info(f"Extracted {len(raw_text)} characters of text from document.")

        # ---------------------------------------------------------------------
        # Step 3: Split text into chunks using LlamaIndex SentenceSplitter
        # ---------------------------------------------------------------------
        logger.info(f"Splitting text into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")

        splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        chunks = splitter.split_text(raw_text)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Text splitting produced no chunks. The document may be empty."
            )

        logger.info(f"Created {len(chunks)} chunks from document.")

        # ---------------------------------------------------------------------
        # Step 4: Generate embeddings for all chunks
        # ---------------------------------------------------------------------
        logger.info("Generating embeddings for all chunks...")

        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            # Generate embedding using the globally configured model
            embedding_vector = Settings.embed_model.get_text_embedding(chunk)

            # Generate unique ID for this chunk
            chunk_id = str(uuid.uuid4())

            # Prepare vector item for Endee
            vectors_to_upsert.append({
                "id": chunk_id,
                "vector": embedding_vector,
                "meta": {
                    "text": chunk,
                    "source_file": file.filename,
                    "chunk_index": i
                }
            })

        logger.info(f"Generated {len(vectors_to_upsert)} embedding vectors.")

        # ---------------------------------------------------------------------
        # Step 5: Ensure index exists and upsert to Endee
        # ---------------------------------------------------------------------
        logger.info("Connecting to Endee index and upserting vectors...")

        try:
            # Try to get the existing index
            index = endee_client.get_index(ENDEE_INDEX_NAME)
            logger.info(f"Connected to existing index '{ENDEE_INDEX_NAME}'.")
        except Exception:
            # Index doesn't exist, create it then get it
            logger.info(f"Index '{ENDEE_INDEX_NAME}' not found. Creating new index...")
            endee_client.create_index(
                name=ENDEE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                space_type="cosine"
            )
            index = endee_client.get_index(ENDEE_INDEX_NAME)
            logger.info(f"Index '{ENDEE_INDEX_NAME}' created successfully.")

        try:
            # Upsert vectors using the Endee client
            index.upsert(vectors_to_upsert)
            successful_upserts = len(vectors_to_upsert)
            failed_upserts = 0
            logger.info(f"Successfully upserted {successful_upserts} vectors to Endee.")

        except Exception as e:
            logger.error(f"Endee upsert failed: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail=f"Failed to upsert vectors to Endee: {str(e)}"
            )

        logger.info(
            f"Ingestion complete: {successful_upserts} chunks stored, "
            f"{failed_upserts} failed."
        )

        # ---------------------------------------------------------------------
        # Step 6: Return success response
        # ---------------------------------------------------------------------
        return {
            "status": "success",
            "message": f"Document '{file.filename}' ingested successfully.",
            "details": {
                "total_chunks": len(chunks),
                "successful_upserts": successful_upserts,
                "failed_upserts": failed_upserts,
                "characters_processed": len(raw_text)
            }
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error during ingestion: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during ingestion: {str(e)}"
        )

    finally:
        # ---------------------------------------------------------------------
        # Cleanup: Remove temporary file
        # ---------------------------------------------------------------------
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except OSError as e:
                logger.warning(f"Failed to clean up temporary file: {str(e)}")


# -----------------------------------------------------------------------------
# RAG Inference Endpoint
# -----------------------------------------------------------------------------
@app.post("/ask", tags=["Inference"])
async def ask_question(request: QueryRequest):
    """
    Performs RAG (Retrieval-Augmented Generation) to answer legal questions.

    Process:
        1. Embed the user's question using local HuggingFace model
        2. Search Endee for the most relevant contract clauses
        3. Extract context from matching vectors
        4. Construct a prompt with the context
        5. Generate answer using Groq LLM

    Args:
        request: QueryRequest containing the user's question

    Returns:
        dict: Contains the generated answer and the context used for citation

    Raises:
        HTTPException: If embedding, search, or LLM generation fails
    """
    try:
        logger.info(f"Received question: {request.query[:100]}...")
        if request.file_name:
            logger.info(f"Filtering to file: {request.file_name}")

        # ---------------------------------------------------------------------
        # Step A: Intent Routing (Greeting Bypass)
        # ---------------------------------------------------------------------
        query_lower = request.query.lower().strip()
        is_greeting = any(
            query_lower == pattern or query_lower.startswith(pattern + " ")
            for pattern in GREETING_PATTERNS
        )

        if is_greeting:
            logger.info("Intent detected: GREETING - Skipping database search.")
            context_string = "No context needed for general greetings."
        else:
            # -----------------------------------------------------------------
            # Step B: Embed the Question
            # -----------------------------------------------------------------
            logger.info("Intent detected: CONTRACT QUERY - Searching database.")
            logger.info("Embedding the user's question...")

            question_vector = Settings.embed_model.get_text_embedding(request.query)

            logger.info(f"Question embedded into {len(question_vector)}-dimensional vector.")

            # -----------------------------------------------------------------
            # Step C: Search Endee for Relevant Clauses (with optional filter)
            # -----------------------------------------------------------------
            try:
                # Connect to the index
                index = endee_client.get_index(ENDEE_INDEX_NAME)

                if request.file_name:
                    # Metadata filtering: fetch more results then filter in Python
                    logger.info(f"Searching Endee with metadata filter for '{request.file_name}'...")
                    results = index.query(vector=question_vector, top_k=RAG_TOP_K_FILTERED)

                    # Filter results to only include chunks from the specified file
                    filtered_results = [
                        r for r in results
                        if r.get("meta", {}).get("source_file") == request.file_name
                    ]

                    # Take only top RAG_TOP_K after filtering
                    results = filtered_results[:RAG_TOP_K]
                    logger.info(f"Filtered to {len(results)} results from '{request.file_name}'.")
                else:
                    # No filter: search all contracts
                    logger.info(f"Searching Endee for top {RAG_TOP_K} relevant clauses (all files)...")
                    results = index.query(vector=question_vector, top_k=RAG_TOP_K)
                    logger.info(f"Endee search returned {len(results)} results.")

            except Exception as e:
                logger.error(f"Endee search failed: {str(e)}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to search vector database: {str(e)}"
                )

            # -----------------------------------------------------------------
            # Step D: Extract Context from Search Results
            # -----------------------------------------------------------------
            logger.info("Extracting context from search results...")

            if not results:
                logger.warning("No relevant clauses found in the vector database.")
                context_string = "No relevant contract clauses were found in the database."
            else:
                # Extract text from the meta field of each matching vector
                context_chunks = []
                for result in results:
                    meta = result.get("meta", {})
                    text = meta.get("text", "")
                    if text:
                        source = meta.get("source_file", "Unknown")
                        context_chunks.append(f"[Source: {source}]\n{text}")

                context_string = "\n\n---\n\n".join(context_chunks)
                logger.info(f"Extracted {len(context_chunks)} context chunks ({len(context_string)} chars).")

        # ---------------------------------------------------------------------
        # Step E: Payload Truncation (Rate Limit Protection)
        # ---------------------------------------------------------------------
        if len(context_string) > CONTEXT_MAX_CHARS:
            logger.warning(f"Context too long ({len(context_string)} chars). Truncating to {CONTEXT_MAX_CHARS}.")
            context_string = context_string[:CONTEXT_MAX_CHARS] + "\n\n... [Context truncated for token limits]"

        # ---------------------------------------------------------------------
        # Step F: Construct the Prompt (Hybrid Agent)
        # ---------------------------------------------------------------------
        logger.info("Constructing prompt for LLM...")

        prompt = f"""You are a highly professional Enterprise Legal AI Assistant.

Follow these strict rules:
- If the user is greeting you, making small talk, or asking about your capabilities, respond naturally and politely as a helpful AI assistant. Do NOT use any internal labels, prefixes, or robotic language.
- If the user asks a question about contracts, agreements, or legal clauses, you must answer using ONLY the context provided below.
- If the answer is not contained in the context, state explicitly: "I do not have enough information in the current database to answer that."

CONTEXT:
{context_string}

USER QUESTION:
{request.query}

RESPONSE:"""

        logger.info(f"Prompt constructed ({len(prompt)} chars).")

        # ---------------------------------------------------------------------
        # Step G: Generate Answer with LLM
        # ---------------------------------------------------------------------
        logger.info("Sending prompt to Groq LLM for generation...")

        try:
            llm_response = Settings.llm.complete(prompt)
            answer = llm_response.text.strip()
            logger.info(f"LLM generated response ({len(answer)} chars).")

        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail=f"Failed to generate answer from LLM: {str(e)}"
            )

        # ---------------------------------------------------------------------
        # Return Response with Answer and Context for Citations
        # ---------------------------------------------------------------------
        return {
            "answer": answer,
            "context_used": context_string
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error during question answering: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )


# -----------------------------------------------------------------------------
# Entry Point for Development
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Enable hot-reload for development
    )
