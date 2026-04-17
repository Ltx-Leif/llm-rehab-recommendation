# external/kb_client.py
import chromadb
from sentence_transformers import SentenceTransformer
from core.config import settings
from utils import get_logger, ExternalToolError
from typing import List, Optional
import torch
import os

logger = get_logger(__name__)

# --- Initialization ---
embed_model = None
chroma_client = None
collection = None

try:
    # Determine device (CPU or CUDA)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"KB Client: Using device: {device}")

    # Load the embedding model (ensure model is downloaded or available)
    logger.info(f"KB Client: Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
    embed_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device=device)
    logger.info(f"KB Client: Embedding model loaded successfully.")

    # Initialize ChromaDB client using the configured persist directory
    persist_directory = settings.CHROMA_PERSIST_DIR
    logger.info(f"KB Client: Attempting to load ChromaDB from persist_directory: {persist_directory}")

    if not os.path.isdir(persist_directory):
         # If the directory itself doesn't exist, ChromaDB cannot load the sqlite file.
         logger.error(f"ChromaDB persist directory does not exist: {persist_directory}. Database cannot be loaded.")
         raise FileNotFoundError(f"ChromaDB persist directory not found: {persist_directory}")
    # Check if the actual sqlite file exists (optional, ChromaDB handles it internally)
    # sqlite_file = os.path.join(persist_directory, 'chroma.sqlite3')
    # if not os.path.exists(sqlite_file):
    #     logger.error(f"ChromaDB database file 'chroma.sqlite3' not found in directory: {persist_directory}")
    #     raise FileNotFoundError(f"ChromaDB database file 'chroma.sqlite3' not found in {persist_directory}")

    # ChromaDB will load from the specified directory if it contains a valid DB structure (like chroma.sqlite3)
    chroma_client = chromadb.PersistentClient(path=persist_directory)

    # Get the existing collection (should not create if DB is pre-built)
    logger.info(f"KB Client: Getting ChromaDB collection: {settings.CHROMA_COLLECTION_NAME}")
    # Use get_collection to ensure it exists, raise error if not found in pre-built DB
    try:
        collection = chroma_client.get_collection(name=settings.CHROMA_COLLECTION_NAME)
        logger.info(f"KB Client: Successfully connected to collection '{settings.CHROMA_COLLECTION_NAME}' with {collection.count()} documents.")
    except Exception as e: # Catch specific ChromaDB exceptions if known, otherwise general Exception
         logger.error(f"KB Client: Failed to get collection '{settings.CHROMA_COLLECTION_NAME}' from DB at '{persist_directory}'. Does the collection exist in the pre-built DB? Error: {e}")
         raise ExternalToolError(f"Failed to get ChromaDB collection '{settings.CHROMA_COLLECTION_NAME}'.", tool_name="KB Client") from e

except ImportError as e:
    logger.error(f"KB Client: Failed to import required libraries (chromadb, sentence_transformers, torch). Install them: {e}")
    raise
except FileNotFoundError as e:
    logger.error(f"KB Client: {e}")
    # Keep components as None, search will fail later
    embed_model = None
    chroma_client = None
    collection = None
    logger.error("KB Client: Knowledge base functionality unavailable due to missing directory/file.")
except Exception as e:
    logger.exception(f"KB Client: Error during initialization: {e}")
    embed_model = None
    chroma_client = None
    collection = None
    logger.error("KB Client: Knowledge base functionality might be unavailable.")
    # Optional: raise ExternalToolError to halt application startup if KB is critical
    # raise ExternalToolError(f"KB Client initialization failed: {e}", tool_name="KB Client") from e


# --- Search Function (Remains the same logic) ---
def search_kb(query_text: str, top_k: int = 3) -> List[str]:
    """
    Searches the ChromaDB knowledge base for relevant documents.

    Args:
        query_text: The text query to search for.
        top_k: The number of top results to return.

    Returns:
        A list of relevant document chunks (strings).

    Raises:
        ExternalToolError: If embedding or ChromaDB query fails, or client not initialized.
    """
    if not embed_model or not collection:
        logger.error("KB Client: Embedding model or ChromaDB collection not initialized. Cannot perform search.")
        raise ExternalToolError("KB Client not initialized.", tool_name="KB Client")

    logger.info(f"KB Client: Searching KB for query (first 100 chars): '{query_text[:100]}...', top_k={top_k}")
    try:
        query_embedding = embed_model.encode([query_text], show_progress_bar=False)
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=['documents']
        )
        documents = results.get("documents", [[]])[0]
        logger.info(f"KB Client: Found {len(documents)} relevant document(s).")
        return documents if documents else []

    except Exception as e:
        logger.exception(f"KB Client: Error during knowledge base search for query '{query_text[:50]}...': {e}")
        raise ExternalToolError(f"KB search failed: {e}", tool_name="KB Client")


# --- Removed Data Loading Functions ---
# _read_md_blocks, process_md_file, process_all_md_files are removed
# as the DB is assumed to be pre-built and located at CHROMA_PERSIST_DIR.

# --- Optional Helper Functions (Keep if useful) ---
# def show_all_documents(): ...
# def delete_document_by_id(doc_id): ...


# Example Usage (for testing module directly)
async def main():
    if not collection:
        print("KB Client failed to initialize or collection not found.")
        return
    try:
        # Test with a query potentially relevant to stroke or tumor rehab
        test_query = "脑卒中后吞咽障碍的康复方法有哪些？"
        print(f"\n--- KB Search Test ---")
        print(f"Query: {test_query}")
        relevant_docs = search_kb(test_query, top_k=3)
        if relevant_docs:
            for i, doc in enumerate(relevant_docs):
                print(f"\nResult {i+1}:\n{doc[:400]}...") # Print slightly longer snippet
        else:
            print("No relevant documents found.")
        print("----------------------\n")

    except ExternalToolError as e:
        print(f"KB Search Test Failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())