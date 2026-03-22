"""
CUAD Batch Ingestion Script
============================
Standalone CLI utility to populate the Endee vector database with the CUAD
(Contract Understanding Atticus Dataset) for building a legal knowledge base.

Usage:
    python batch_ingest.py

Prerequisites:
    - Endee Docker container running on localhost:8080
    - CUAD dataset extracted to ./cuad_data folder
    - NVIDIA GPU with CUDA support

Note: This script ONLY processes PDF files. All other file types
(TXT, JSON, Excel, etc.) are explicitly ignored.

OCR Note: The CUAD dataset PDFs already contain clean, extracted text
(OCR has been pre-applied). We use the default 'auto' strategy.
To enable hi_res OCR for scanned/image PDFs, uncomment the strategy parameter.
"""

import os
import sys
import time
import uuid
import logging
import traceback
from pathlib import Path

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from unstructured.partition.auto import partition

from endee import Endee

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
CUAD_DIR = Path("./cuad_data")

# Endee Configuration
ENDEE_BASE_URL = "http://localhost:8080/api/v1"
ENDEE_INDEX_NAME = "contracts"

# Embedding Configuration
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMENSION = 1024

# Chunking Configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Processing Configuration
DELAY_BETWEEN_FILES = 0.5  # seconds

# =============================================================================
# PDF Parsing Strategy
# =============================================================================
# The CUAD dataset PDFs already have clean, selectable text (pre-OCR'd).
# We use the default 'auto' strategy which is faster for text-based PDFs.
#
# UNCOMMENT THE LINE BELOW to enable hi_res OCR for scanned/image-based PDFs:
# PDF_STRATEGY = "hi_res"  # Requires: unstructured[pdf] + unstructured-inference
# =============================================================================


def initialize_endee_client() -> Endee:
    """Initialize and return the Endee client."""
    logger.info("Initializing Endee client...")
    client = Endee(token="local-dev-token")
    client.set_base_url(ENDEE_BASE_URL)
    logger.info(f"Endee client configured for: {ENDEE_BASE_URL}")
    return client


def initialize_embedding_model() -> HuggingFaceEmbedding:
    """Initialize and return the HuggingFace embedding model on GPU."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    logger.info("This will use your local GPU (CUDA)...")

    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        device="cuda",
        trust_remote_code=True
    )

    # Set as global LlamaIndex setting
    Settings.embed_model = embed_model

    logger.info("Embedding model loaded successfully on GPU.")
    return embed_model


def ensure_index_exists(client: Endee) -> None:
    """Ensure the contracts index exists in Endee, create if not."""
    logger.info(f"Checking for index '{ENDEE_INDEX_NAME}'...")

    try:
        client.get_index(ENDEE_INDEX_NAME)
        logger.info(f"Index '{ENDEE_INDEX_NAME}' already exists.")
    except Exception:
        logger.info(f"Index '{ENDEE_INDEX_NAME}' not found. Creating...")
        client.create_index(
            name=ENDEE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            space_type="cosine"
        )
        logger.info(f"Index '{ENDEE_INDEX_NAME}' created successfully.")


def get_pdf_files(directory: Path) -> list[Path]:
    """
    Find all PDF files in the directory.

    Strictly filters to only .pdf extension.
    Ignores TXT, JSON, Excel, and all other file types.
    """
    pdf_files = []

    for file_path in directory.rglob("*.pdf"):
        if file_path.is_file():
            pdf_files.append(file_path)

    # Sort for consistent processing order
    pdf_files.sort()

    return pdf_files


def process_single_pdf(
    file_path: Path,
    splitter: SentenceSplitter,
    embed_model: HuggingFaceEmbedding,
    index
) -> tuple[int, int]:
    """
    Process a single PDF file: extract text via OCR, chunk, embed, and upsert.

    Uses hi_res strategy for OCR-based extraction from scanned/image PDFs.

    Returns:
        Tuple of (chunks_created, vectors_upserted)

    Raises:
        Exception: If PDF parsing, embedding, or upsert fails
    """
    # -------------------------------------------------------------------------
    # Extract text using unstructured
    # -------------------------------------------------------------------------
    # CUAD dataset PDFs have clean, selectable text - use default 'auto' strategy
    # For scanned/image PDFs, uncomment the strategy parameter below:
    # -------------------------------------------------------------------------
    elements = partition(
        filename=str(file_path),
        # strategy="hi_res",  # UNCOMMENT FOR OCR (scanned/image PDFs)
    )

    if not elements:
        logger.warning(f"  No text extracted from {file_path.name}")
        return 0, 0

    raw_text = "\n\n".join([str(element) for element in elements])

    if not raw_text.strip():
        logger.warning(f"  Empty text content in {file_path.name}")
        return 0, 0

    logger.info(f"  Extracted {len(raw_text):,} characters")

    # Split into chunks
    chunks = splitter.split_text(raw_text)

    if not chunks:
        logger.warning(f"  No chunks created from {file_path.name}")
        return 0, 0

    logger.info(f"  Split into {len(chunks)} chunks, generating embeddings...")

    # Generate embeddings and prepare vectors
    vectors_to_upsert = []

    for i, chunk in enumerate(chunks):
        embedding_vector = embed_model.get_text_embedding(chunk)
        chunk_id = str(uuid.uuid4())

        vectors_to_upsert.append({
            "id": chunk_id,
            "vector": embedding_vector,
            "meta": {
                "text": chunk,
                "source_file": file_path.name,
                "chunk_index": i
            }
        })

    # Upsert to Endee
    logger.info(f"  Upserting {len(vectors_to_upsert)} vectors to Endee...")
    index.upsert(vectors_to_upsert)

    return len(chunks), len(vectors_to_upsert)


def main():
    """Main entry point for batch PDF ingestion."""
    logger.info("=" * 70)
    logger.info("CUAD Batch Ingestion Script")
    logger.info("Strategy: auto (CUAD PDFs have clean, pre-extracted text)")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Validate CUAD directory exists
    # -------------------------------------------------------------------------
    if not CUAD_DIR.exists():
        logger.error(f"CUAD directory not found: {CUAD_DIR.absolute()}")
        logger.error("Please download the CUAD dataset and extract it to ./cuad_data")
        sys.exit(1)

    if not CUAD_DIR.is_dir():
        logger.error(f"CUAD path is not a directory: {CUAD_DIR.absolute()}")
        sys.exit(1)

    logger.info(f"CUAD directory found: {CUAD_DIR.absolute()}")

    # -------------------------------------------------------------------------
    # Step 2: Initialize components
    # -------------------------------------------------------------------------
    endee_client = initialize_endee_client()
    embed_model = initialize_embedding_model()

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    logger.info(f"Sentence splitter configured (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # -------------------------------------------------------------------------
    # Step 3: Ensure index exists
    # -------------------------------------------------------------------------
    ensure_index_exists(endee_client)
    index = endee_client.get_index(ENDEE_INDEX_NAME)

    # -------------------------------------------------------------------------
    # Step 4: Discover PDF files only
    # -------------------------------------------------------------------------
    logger.info(f"Scanning for PDF files in {CUAD_DIR}...")
    pdf_files = get_pdf_files(CUAD_DIR)

    if not pdf_files:
        logger.warning("No PDF files found in the CUAD directory.")
        logger.warning("Ensure the dataset contains .pdf files.")
        sys.exit(0)

    total_files = len(pdf_files)
    logger.info(f"Found {total_files} PDF files to process")
    logger.info("-" * 70)

    # -------------------------------------------------------------------------
    # Step 5: Process each PDF file
    # -------------------------------------------------------------------------
    total_chunks = 0
    total_vectors = 0
    successful_files = 0
    failed_files = 0
    failed_file_names = []

    start_time = time.time()

    for i, file_path in enumerate(pdf_files, start=1):
        progress_pct = (i / total_files) * 100
        logger.info(f"Processing {i}/{total_files} ({progress_pct:.1f}%): {file_path.name}")

        try:
            chunks, vectors = process_single_pdf(
                file_path=file_path,
                splitter=splitter,
                embed_model=embed_model,
                index=index
            )

            if vectors > 0:
                total_chunks += chunks
                total_vectors += vectors
                successful_files += 1
                logger.info(f"  -> SUCCESS: {chunks} chunks, {vectors} vectors upserted")
            else:
                logger.warning(f"  -> SKIPPED: No content extracted (possibly corrupted)")
                failed_files += 1
                failed_file_names.append(file_path.name)

        except KeyboardInterrupt:
            logger.warning("\nInterrupted by user. Saving progress summary...")
            break

        except Exception as e:
            # Catch ANY exception to prevent batch process from crashing
            error_type = type(e).__name__
            logger.error(f"  -> FAILED [{error_type}]: {str(e)}")
            logger.debug(f"  Traceback: {traceback.format_exc()}")
            failed_files += 1
            failed_file_names.append(f"{file_path.name} ({error_type})")
            # Continue to next file - do not crash the batch
            continue

        # Delay between files to prevent overwhelming the system
        if i < total_files:
            time.sleep(DELAY_BETWEEN_FILES)

    # -------------------------------------------------------------------------
    # Step 6: Print summary
    # -------------------------------------------------------------------------
    elapsed_time = time.time() - start_time
    files_processed = successful_files + failed_files

    logger.info("=" * 70)
    logger.info("BATCH INGESTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"PDF Strategy:           auto (pre-extracted text)")
    logger.info(f"Total PDFs found:       {total_files}")
    logger.info(f"PDFs processed:         {files_processed}")
    logger.info(f"Successful:             {successful_files}")
    logger.info(f"Failed/Skipped:         {failed_files}")
    logger.info(f"Total chunks created:   {total_chunks:,}")
    logger.info(f"Total vectors upserted: {total_vectors:,}")
    logger.info(f"Time elapsed:           {elapsed_time:.2f} seconds")

    if files_processed > 0:
        logger.info(f"Average per PDF:        {elapsed_time / files_processed:.2f} seconds")

    if failed_file_names:
        logger.info("-" * 70)
        logger.info(f"Failed/Skipped files ({len(failed_file_names)} total):")
        for name in failed_file_names[:25]:  # Show first 25 failures
            logger.info(f"  - {name}")
        if len(failed_file_names) > 25:
            logger.info(f"  ... and {len(failed_file_names) - 25} more")

    logger.info("=" * 70)

    # Return exit code based on success rate
    if successful_files == 0 and total_files > 0:
        sys.exit(1)  # Complete failure
    elif failed_files > 0:
        sys.exit(0)  # Partial success (some failures are expected with OCR)
    else:
        sys.exit(0)  # Full success


if __name__ == "__main__":
    main()
