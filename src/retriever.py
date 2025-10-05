import os
import logging
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('retriever')

# Load environment variables from .env file
load_dotenv()

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX")

# Initialize search client only if configuration is valid
def get_search_client():
    logger.debug("Initializing search client")
    if None not in [SEARCH_ENDPOINT, SEARCH_KEY, INDEX_NAME] and "<your-search>" not in SEARCH_ENDPOINT and "<your-key>" not in SEARCH_KEY:
        logger.debug(f"Creating search client with endpoint {SEARCH_ENDPOINT} and index {INDEX_NAME}")
        return SearchClient(endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=AzureKeyCredential(SEARCH_KEY))
    logger.error("Invalid search configuration")
    return None

search_client = get_search_client()

def search_similar(query_vector, k=5):
    if search_client is None:
        logger.error("Search client not initialized. Check your Azure Search configuration in .env file.")
        return []
    
    # Use the exact field name from the index schema (embedding_vector)
    vector_query_options = [
        # Default configuration with embedding_vector field and default-profile
        {
            "vector": {
                "value": query_vector,
                "fields": "embedding_vector",
                "k": k,
                "profile": "default-profile"
            }
        },
        # Fallback without profile specification
        {
            "vector": {
                "value": query_vector,
                "fields": "embedding_vector",
                "k": k
            }
        }
    ]
    
    # Try each vector query configuration
    for i, vector_query in enumerate(vector_query_options):
        try:
            logger.debug(f"Trying vector search configuration {i+1}...")
            results = search_client.search(search_text="", vector=vector_query)
            result_list = [r for r in results]
            if result_list:
                logger.info(f"Vector search successful with configuration {i+1}")
                return result_list
            logger.debug(f"No results found with configuration {i+1}")
        except Exception as e:
            logger.warning(f"Error with vector search configuration {i+1}: {str(e)}")
    
    # Fallback to basic search if all vector searches fail
    try:
        logger.warning("Falling back to basic keyword search...")
        results = search_client.search(search_text="*", top=k)
        return [r for r in results]
    except Exception as e2:
        logger.error(f"Error performing fallback search: {str(e2)}")
        return []

# Add search_similar_chunks function that uses embedder to convert input text to vectors
from embedder import embed_texts

def search_similar_chunks(query_input, top_k=5, k=None):
    """
    Search for chunks similar to the query text or embedding vector.
    
    Args:
        query_input: Either a query text string or query embedding vector
        top_k: Number of results to return (default parameter)
        k: Alternative parameter name for top_k for compatibility
        
    Returns:
        List of search results
    """
    # Use k parameter if provided, otherwise use top_k
    if k is not None:
        top_k = k
    
    try:
        # Check if input is already a vector (list/array) or a text string
        if isinstance(query_input, (list, tuple)) or hasattr(query_input, 'shape'):
            # Input is already a vector
            query_vector = query_input
            logger.info(f"Using provided query embedding vector with {len(query_vector)} dimensions")
        else:
            # Input is text, generate embedding
            query_text = str(query_input)
            logger.info(f"Searching for chunks similar to: '{query_text[:50]}...'")
            query_vectors = embed_texts([query_text])
            
            if not query_vectors:
                logger.error("Failed to generate embedding for query")
                return []
                
            query_vector = query_vectors[0]
            logger.debug(f"Generated query embedding with {len(query_vector)} dimensions")
        
        # Perform vector search
        logger.info(f"Performing vector search with top_k={top_k}")
        results = search_similar(query_vector, k=top_k)
        
        logger.info(f"Found {len(results)} similar chunks")
        return results
        
    except Exception as e:
        logger.error(f"Error searching for similar chunks: {str(e)}")
        return []
