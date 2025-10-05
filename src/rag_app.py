#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main RAG Application Script

This script implements a complete RAG (Retrieval Augmented Generation) pipeline:
1. Chunk text files using the advanced chunking logic from chunk_text_for_rags.py
2. Check if chunks already exist in Azure Search
3. If not, generate embeddings using Azure OpenAI and index them in Azure Search
4. Accept user questions and use the RAG pipeline to find relevant chunks and answer

Usage:
    python rag_app.py --input ../data --question "What is the main topic of the document?"
"""

import os
import sys
import json
import logging
import argparse
import time
import re
import base64
from typing import List, Dict, Any, Optional, Iterable
from dotenv import load_dotenv

# Import components from existing files
from chunk_text_for_rags import process_files_and_generate_records
from embedder import embed_texts, get_openai_client
from indexer import get_search_client
from retriever import search_similar_chunks

def sha1(text: str) -> str:
    """
    Create a SHA1 hash of a string.
    
    Args:
        text: The text to hash
        
    Returns:
        SHA1 hash of the text
    """
    import hashlib
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def get_content_hash_id(content: str, prefix: str = "doc_") -> str:
    """
    Generate a document ID using the SHA1 hash of the content.
    This ensures IDs are unique for each distinct content and valid for Azure Search.
    
    Args:
        content: The document content to hash
        prefix: Optional prefix for the ID
        
    Returns:
        A document ID based on content hash
    """
    # Create a hash of the content
    content_hash = sha1(content)
    
    # Return prefixed hash as ID (all valid characters for Azure Search)
    return f"{prefix}{content_hash}"

def sanitize_id(doc_id: str, content: str = None) -> str:
    """
    Sanitize document ID for Azure Search or generate a new ID based on content hash.
    
    Azure Search only allows letters, digits, underscore (_), dash (-), or equal sign (=).
    
    Args:
        doc_id: Original document ID
        content: If provided, use this to generate a hash-based ID instead
        
    Returns:
        A sanitized document ID valid for Azure Search
    """
    # If content is provided, use its hash as the ID
    if content:
        return get_content_hash_id(content)
    
    # Otherwise sanitize the existing ID
    sanitized = re.sub(r'[^a-zA-Z0-9_\-=]', '_', doc_id)
    
    # If the ID structure is too mangled after sanitization, use base64 encoding
    if '::' in doc_id or sanitized.count('_') > 5:
        # Use URL-safe base64 encoding (uses only letters, numbers, '-', and '_')
        encoded = base64.urlsafe_b64encode(doc_id.encode()).decode()
        return encoded
    
    return sanitized

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('rag_app')

# Load environment variables
load_dotenv()

# Get configuration from environment variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
CHAT_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo")
EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")
SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

# Initialize clients
search_client = get_search_client()
openai_client = get_openai_client()

def check_chunks_exist(chunk_ids: List[str]) -> bool:
    """
    Check if the chunks with the given IDs already exist in Azure Search
    
    Args:
        chunk_ids: List of chunk IDs to check
        
    Returns:
        True if all chunks exist, False otherwise
    """
    if not search_client:
        logger.error("Search client not initialized")
        return False
    
    # Batch chunks to avoid hitting API limits
    batch_size = 50
    for i in range(0, len(chunk_ids), batch_size):
        batch = chunk_ids[i:i+batch_size]
        logger.info(f"Checking if {len(batch)} chunks exist in Azure Search (batch {i//batch_size + 1})")
        
        # Use filter to check for existence
        filter_str = " or ".join([f"id eq '{id}'" for id in batch])
        results = search_client.search("", filter=filter_str, select=["id"], top=len(batch))
        
        found_ids = set(doc["id"] for doc in results)
        if len(found_ids) < len(batch):
            logger.info(f"Not all chunks found in Azure Search: {len(found_ids)} of {len(batch)}")
            return False
    
    logger.info(f"All {len(chunk_ids)} chunks already exist in Azure Search")
    return True

def check_chunks_exist_by_hash(chunks: List[Dict[str, Any]]) -> bool:
    """
    Check if chunks already exist in the Azure Search index based on content hash
    
    Args:
        chunks: List of chunk documents with content to check
        
    Returns:
        True if all chunks exist (by content hash), False otherwise
    """
    if not search_client:
        logger.error("Search client not initialized")
        return False
    
    # Generate content hash IDs for all chunks
    hash_ids = [get_content_hash_id(chunk["content"]) for chunk in chunks]
    logger.info(f"Checking if {len(hash_ids)} chunks exist in Azure Search by content hash")
    
    # Batch check to avoid hitting API limits
    batch_size = 50
    for i in range(0, len(hash_ids), batch_size):
        batch = hash_ids[i:i+batch_size]
        logger.info(f"Checking batch {i//batch_size + 1} with {len(batch)} hash IDs")
        
        # Use filter to check for existence
        filter_str = " or ".join([f"id eq '{id}'" for id in batch])
        results = search_client.search("", filter=filter_str, select=["id"], top=len(batch))
        
        found_ids = set(doc["id"] for doc in results)
        if len(found_ids) < len(batch):
            missing = set(batch) - found_ids
            logger.info(f"Not all chunks found by hash: {len(found_ids)} of {len(batch)} exist")
            logger.debug(f"Missing hash IDs: {missing}")
            return False
    
    logger.info(f"All {len(hash_ids)} chunks already exist in Azure Search (matched by content hash)")
    return True

def embed_and_index_chunks(chunks: List[Dict[str, Any]]) -> None:
    """
    Generate embeddings for chunks and index them in Azure Search
    
    Args:
        chunks: List of chunk documents with 'id' and 'content' fields
    """
    logger.info(f"Generating embeddings for {len(chunks)} chunks")
    
    # First, prepare documents to match Azure Search schema
    # The schema allows only these fields: id, content, metadata, embedding_vector
    logger.info("Preparing documents for Azure Search schema")
    
    # Process in batches to avoid memory issues and respect API limits
    batch_size = 16
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} chunks)")
        
        # Extract content for embedding
        texts = [doc["content"] for doc in batch]
        
        # Generate embeddings
        try:
            embeddings = embed_texts(texts)
            
            # Prepare documents that exactly match the schema
            schema_docs = []
            for j, embedding in enumerate(embeddings):
                doc = batch[j]
                
                # Create a new document with only the allowed fields
                # Use content hash for the document ID to ensure uniqueness and valid characters
                content = doc["content"]
                content_hash_id = get_content_hash_id(content)
                
                schema_doc = {
                    "id": content_hash_id,  # Use hash of content as ID
                    "content": content,
                    "embedding_vector": embedding
                }
                
                # Handle metadata - must be a string according to schema
                metadata_dict = {}
                
                # Include original source path if available
                if "source_path" in doc:
                    metadata_dict["source_path"] = doc["source_path"]
                
                # Include filename and extension if available in original metadata
                if "metadata" in doc and isinstance(doc["metadata"], dict):
                    if "filename" in doc["metadata"]:
                        metadata_dict["filename"] = doc["metadata"]["filename"]
                    if "ext" in doc["metadata"]:
                        metadata_dict["ext"] = doc["metadata"]["ext"]
                
                # Convert metadata dict to JSON string as required by the schema
                schema_doc["metadata"] = json.dumps(metadata_dict)
                
                schema_docs.append(schema_doc)
                logger.debug(f"Prepared document {doc['id']} according to schema")
            
            # Index the batch with schema-compliant documents
            if search_client:
                results = search_client.upload_documents(documents=schema_docs)
                success_count = sum(1 for r in results if r.succeeded)
                logger.info(f"Indexed {success_count}/{len(batch)} chunks in Azure Search")
            else:
                logger.error("Search client not initialized")
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}", exc_info=True)

def query_azureopenai(messages: List[Dict[str, str]]) -> str:
    """
    Query the chat model with the given messages
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        
    Returns:
        The model's response as a string
    """
    try:
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error querying chat model: {str(e)}")
        return f"I'm sorry, I encountered an error: {str(e)}"

def build_prompt(retrieved_chunks: List[Dict], user_question: str) -> List[Dict[str, str]]:
    """
    Build a prompt for the chat model using retrieved chunks
    
    Args:
        retrieved_chunks: List of chunk documents
        user_question: The user's question
        
    Returns:
        A list of message dictionaries for the chat model
    """
    # Format the context from retrieved chunks
    context_parts = []
    
    for i, chunk in enumerate(retrieved_chunks):
        # Extract metadata if it exists and is in JSON format
        metadata_info = "unknown source"
        if "metadata" in chunk and isinstance(chunk["metadata"], str):
            try:
                metadata_dict = json.loads(chunk["metadata"])
                if "filename" in metadata_dict:
                    metadata_info = f"Document: {metadata_dict['filename']}"
            except json.JSONDecodeError:
                # If metadata is not valid JSON, use it directly
                metadata_info = f"Source: {chunk['metadata']}"
        
        context_parts.append(f"[{i+1}] {metadata_info}\n{chunk['content']}\n")
    
    context = "\n".join(context_parts)
    
    # Build the system and user messages
    system_message = (
        "You are a helpful AI assistant. Answer the question based ONLY on the provided context. "
        "If the answer cannot be determined from the context, say 'I don't have enough information to answer that question.' "
        "Include references to the relevant document numbers in your answer."
    )
    
    user_message = f"Context information:\n\n{context}\n\nQuestion: {user_question}"
    
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

def process_and_index_data(input_path: str, max_tokens: int = 512, overlap: int = 50, chunk_output: Optional[str] = None) -> List[str]:
    """
    Process text files, generate chunks, and index them if needed
    
    Args:
        input_path: Path to input directory or file
        max_tokens: Maximum tokens per chunk
        overlap: Token overlap between chunks
        chunk_output: Path to save chunks as JSONL (optional)
        
    Returns:
        List of chunk IDs
    """
    # Resolve paths
    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)
    
    logger.info(f"Processing input from: {input_path}")
    
    # Process files and generate chunks
    if os.path.isdir(input_path):
        paths_to_process = [os.path.join(root, f) 
                           for root, _, files in os.walk(input_path) 
                           for f in files if f.lower().endswith('.txt')]
    elif os.path.isfile(input_path) and input_path.lower().endswith('.txt'):
        paths_to_process = [input_path]
    else:
        logger.error(f"Input path is not a valid directory or .txt file: {input_path}")
        sys.exit(1)
    
    # Generate chunk records from files
    logger.info(f"Found {len(paths_to_process)} text files to process")
    
    # Create records generator
    chunk_generator = process_files_and_generate_records(paths_to_process, max_tokens, overlap)
    
    # Load all chunks (we need them in memory for processing)
    chunks = list(chunk_generator)
    logger.info(f"Generated {len(chunks)} chunks from the input files")
    
    # Transform chunks to match Azure Search index requirements
    for chunk in chunks:
        # Add source info to content for better context
        if "metadata" in chunk and isinstance(chunk["metadata"], dict):
            metadata = chunk["metadata"]
            # Add source info to content for better context
            source_info = f"Source: {metadata.get('filename', 'unknown')}"
            chunk["content"] = f"{source_info}\n\n{chunk['content']}"
            
            # Keep metadata as is - it will be properly formatted during indexing
    
    # Write chunks to JSONL file if requested
    if chunk_output:
        with open(chunk_output, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        logger.info(f"Saved chunks to {chunk_output}")
    
    # Check if chunks exist in the index using content hash
    if not check_chunks_exist_by_hash(chunks):
        logger.info("Some chunks not found in Azure Search, embedding and indexing...")
        embed_and_index_chunks(chunks)
    else:
        logger.info("All chunks already exist in Azure Search (matched by content hash), skipping embedding and indexing")
    
    # Generate and return hash-based IDs for all chunks
    hash_ids = [get_content_hash_id(chunk["content"]) for chunk in chunks]
    return hash_ids

def interactive_mode():
    """Run an interactive question-answering loop"""
    print("\n🔍 RAG Interactive Query Mode\n")
    print("Type your questions below or type 'exit' to quit.\n")
    
    while True:
        try:
            question = input("\nYour question: ")
            if question.lower() in ['exit', 'quit', 'q']:
                break
            
            if not question.strip():
                continue
            
            # Embed the question
            question_embedding = embed_texts([question])[0]
            
            # Search for similar chunks
            retrieved_chunks = search_similar_chunks(question_embedding, k=10)
            if not retrieved_chunks:
                print("No relevant information found in the index.")
                continue
            
            print(f"\nFound {len(retrieved_chunks)} relevant chunks...")
            
            # Build prompt and query the model
            messages = build_prompt(retrieved_chunks, question)
            
            print("\nGenerating answer...")
            answer = query_azureopenai(messages)
            
            # Display the answer
            print("\n" + "="*80)
            print("📝 ANSWER:\n")
            print(answer)
            print("="*80)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="RAG Application")
    parser.add_argument("--input", help="Path to input directory or file with .txt files")
    parser.add_argument("--question", help="Question to answer (if not provided, enters interactive mode)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per chunk")
    parser.add_argument("--overlap", type=int, default=50, help="Token overlap between chunks")
    parser.add_argument("--save-chunks", help="Path to save chunks as JSONL file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate configuration
    if not openai_client:
        logger.error("Azure OpenAI client could not be initialized. Check your .env file.")
        sys.exit(1)
    
    if not search_client:
        logger.error("Azure Search client could not be initialized. Check your .env file.")
        sys.exit(1)
    
    # Process input files if provided
    if args.input:
        process_and_index_data(args.input, args.max_tokens, args.overlap, args.save_chunks)
    
    # Handle question or enter interactive mode
    if args.question:
        # Single question mode
        question = args.question
        
        # Embed the question
        question_embedding = embed_texts([question])[0]
        
        # Search for similar chunks
        retrieved_chunks = search_similar_chunks(question_embedding, k=10)
        if not retrieved_chunks:
            print("No relevant information found in the index.")
            sys.exit(0)
        
        print(f"Found {len(retrieved_chunks)} relevant chunks...")
        
        # Build prompt and query the model
        messages = build_prompt(retrieved_chunks, question)
        
        print("Generating answer...")
        answer = query_azureopenai(messages)
        
        # Display the answer
        print("\n" + "="*80)
        print("📝 ANSWER:\n")
        print(answer)
        print("="*80)
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess terminated by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)