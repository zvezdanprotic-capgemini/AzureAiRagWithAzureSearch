import os
import logging
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('indexer')

# Load environment variables from .env file
load_dotenv()

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX")

logger.debug(f"Azure Search Endpoint: {SEARCH_ENDPOINT}")
logger.debug(f"Azure Search Index: {INDEX_NAME}")

# Initialize search client only if configuration is valid
def get_search_client():
    logger.debug("Initializing Azure Search client")
    if None in [SEARCH_ENDPOINT, SEARCH_KEY, INDEX_NAME]:
        logger.error("Missing Azure Search configuration parameters")
        return None
        
    if "<your-search>" in SEARCH_ENDPOINT or "<your-key>" in SEARCH_KEY:
        logger.error("Azure Search configuration contains placeholder values")
        return None
        
    logger.info(f"Creating search client for index '{INDEX_NAME}'")
    return SearchClient(endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=AzureKeyCredential(SEARCH_KEY))

search_client = get_search_client()
if search_client:
    logger.info("Azure Search client initialized successfully")
else:
    logger.warning("Azure Search client not initialized")

def index_chunks(chunks, vectors, metadata):
    """Index text chunks and their embedding vectors in Azure Search"""
    if search_client is None:
        logger.error("Search client not initialized. Check your Azure Search configuration in .env file.")
        return []
    
    logger.info(f"Preparing to index {len(chunks)} chunks")
    
    docs = []
    for i, (text, vector) in enumerate(zip(chunks, vectors)):
        # Prepare metadata as a JSON string as required by new schema
        if isinstance(metadata, dict):
            metadata_str = str(metadata)
        else:
            metadata_str = metadata
        
        # Create unique ID based on metadata
        doc_id = f"{metadata.get('id','doc')}_{i}"
        logger.debug(f"Creating document with ID: {doc_id}")
        
        # Create document with fields matching the schema
        doc = {
            "id": doc_id,
            "content": text,
            "metadata": metadata_str,
            "embedding": vector,  # Try with the field name 'embedding' first
            "categoryId": "doc"   # Add categoryId field required by some schemas
        }
        docs.append(doc)
    
    # Try multiple field name variations to handle different schemas
    field_variations = [
        {"field": "embedding", "convert": False},
        {"field": "embedding_vector", "convert": False},
        {"field": "vector", "convert": False},
        {"field": "embedding", "convert": True},  # Try converting to list
        {"field": "embedding_vector", "convert": True}
    ]
    
    for variation in field_variations:
        field_name = variation["field"]
        convert = variation["convert"]
        
        try:
            # Update field name in all documents
            for doc in docs:
                # Remove any previous embedding field
                for field in ["embedding", "embedding_vector", "vector"]:
                    if field in doc and field != field_name:
                        del doc[field]
                
                # Add the embedding with the current field name
                vector_value = doc.get("embedding", doc.get("embedding_vector", doc.get("vector", [])))
                if convert and hasattr(vector_value, 'tolist'):
                    doc[field_name] = vector_value.tolist()
                else:
                    doc[field_name] = vector_value
            
            logger.info(f"Trying to upload documents with field name: {field_name} (convert={convert})")
            result = search_client.upload_documents(documents=docs)
            logger.info(f"Successfully indexed {sum(1 for r in result if r.succeeded)} documents using field: {field_name}")
            return result
        except Exception as e:
            logger.warning(f"Failed with field name {field_name}: {str(e)}")
            # Continue to the next variation
    
    # If we're here, all attempts failed
    logger.error("All indexing attempts failed")
    return []
