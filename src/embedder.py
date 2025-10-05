import os
import logging
from dotenv import load_dotenv
from openai import AzureOpenAI
import time

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('embedder')

# Load environment variables from .env file
load_dotenv()

# Get Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")

# Log configuration at module load time
logger.debug(f"Azure OpenAI Endpoint: {AZURE_OPENAI_ENDPOINT}")
logger.debug(f"Embedding Model: {EMBED_MODEL}")

# Initialize Azure OpenAI client
def get_openai_client():
    """Create an Azure OpenAI client"""
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,  
        api_version="2023-05-15",
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )

def embed_texts(texts, model=EMBED_MODEL, batch_size=16):
    """
    Generate embeddings for a list of texts using Azure OpenAI
    
    Args:
        texts (list): List of text strings to embed
        model (str): The deployment name of the embedding model
        batch_size (int): Number of texts to process in one batch
        
    Returns:
        list: List of embedding vectors corresponding to input texts
    """
    # Remove empty texts to avoid API errors
    texts = [text for text in texts if text and text.strip()]
    if not texts:
        logger.warning("No valid texts to embed")
        return []
    
    # Get Azure OpenAI client
    client = get_openai_client()
    vectors = []
    logger.info(f"Embedding {len(texts)} texts with batch size {batch_size}")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        logger.debug(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} with {len(batch)} texts")
        
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Create embeddings for the batch using new Azure OpenAI client
                logger.debug(f"Calling Azure OpenAI embeddings API with model: {model}")
                response = client.embeddings.create(
                    input=batch,
                    model=model  # Use 'model' parameter for new client
                )
                
                # Extract embedding vectors
                batch_vectors = [item.embedding for item in response.data]
                vectors.extend(batch_vectors)
                
                logger.info(f"Successfully embedded batch {i//batch_size + 1}")
                break  # Success, exit retry loop
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"Error embedding batch (attempt {retry_count}/{max_retries}): {str(e)}")
                
                if retry_count < max_retries:
                    # Exponential backoff
                    sleep_time = 2 ** retry_count
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to embed batch after {max_retries} attempts")
                    raise
    
    logger.info(f"Completed embedding all {len(texts)} texts, generated {len(vectors)} vectors")
    return vectors
