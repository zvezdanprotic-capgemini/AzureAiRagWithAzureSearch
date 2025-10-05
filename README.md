# Enhanced RAG System with Azure OpenAI + Azure AI Search

This repository provides a robust implementation of a Retrieval-Augmented Generation (RAG) system using:
- **Azure OpenAI** for embeddings and text generation
- **Azure AI Search** for vector search and retrieval

## Key Features
- Content hash-based document IDs for reliable deduplication
- Schema-compliant document indexing with Azure Search
- Robust error handling and retries with exponential backoff
- Advanced chunking strategies for optimal retrieval
- Comprehensive logging throughout the pipeline
- Interactive query mode for easy testing

## Project Structure
- `src/chunk_text_for_rags.py`: Advanced text chunking utilities with semantic boundaries
- `src/embedder.py`: Embedding generation via Azure OpenAI
- `src/indexer.py`: Document indexing to Azure AI Search
- `src/retriever.py`: Vector similarity search against Azure Search
- `src/rag_app.py`: Main application with processing and interactive query modes

## Setup

### Prerequisites
1. Azure OpenAI service with deployed embedding and chat models
2. Azure AI Search service with a vector-enabled index
3. Python 3.8+ with pip and venv

### Environment Setup
1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration
Copy `.env.example` to `.env` in the root directory and update with your Azure resources:

```bash
cp .env.example .env
# Then edit .env with your actual values
```

Required environment variables:
```
# Azure AI Foundry
AZURE_AI_INFERENCE_ENDPOINT="https://<your-resource-name>-<your-region>.cognitiveservices.azure.com/"
AZURE_AI_EMBED_MODEL="text-embedding-3-small"
AZURE_AI_CHAT_MODEL="gpt-4o-mini"

# Azure AI Search
AZURE_SEARCH_ENDPOINT="https://<your-search-name>.search.windows.net"
AZURE_SEARCH_ADMIN_KEY="<your-search-admin-key>"
AZURE_SEARCH_INDEX="my-rag-index"

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY="<your-openai-api-key>"
AZURE_OPENAI_ENDPOINT="https://<your-resource-name>-<your-region>.cognitiveservices.azure.com/"
AZURE_OPENAI_API_VERSION="2024-12-01-preview"
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o-mini"
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME="text-embedding-3-small"
AZURE_OPENAI_EMBEDDINGS_API_VERSION="2024-12-01-preview"
AZURE_OPENAI_TEMPERATURE="1.0"
```

## Using the System

### 1. Index Your Documents
Use the main RAG application to process and index your documents:

```bash
# Process text files in a directory and index them in Azure Search
python src/rag_app.py --input ../data --save-chunks ./out/chunks.jsonl
```

Options:
- `--input PATH`: Path to a directory or file with text documents to process
- `--save-chunks PATH`: Optional path to save the generated chunks as JSONL
- `--max-tokens NUM`: Maximum tokens per chunk (default: 512)
- `--overlap NUM`: Token overlap between chunks (default: 50)

### 2. Query Your Knowledge Base
Use the interactive query mode:

```bash
# Interactive mode (ask questions)
python src/rag_app.py

# Direct question mode
python src/rag_app.py --question "What is vector search?"
```

### 3. Using with Virtual Environment
Always activate the virtual environment before running:

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
cd src
python rag_app.py
```

## Azure Search Index Schema

The system requires the following Azure AI Search index schema (defined in index_schema.json):

```json
{
  "name": "my-rag-index",
  "fields": [
    { "name": "id", "type": "Edm.String", "key": true, "filterable": true },
    { "name": "content", "type": "Edm.String", "searchable": true, "retrievable": true },
    { "name": "metadata", "type": "Edm.String", "searchable": true, "retrievable": true, "filterable": true },
    {
      "name": "embedding_vector",
      "type": "Collection(Edm.Single)",
      "searchable": true,
      "retrievable": false,
      "stored": false,
      "dimensions": 1536,
      "vectorSearchProfile": "default-profile"
    }
  ],
  "vectorSearch": {
    "algorithms": [
      {
        "name": "hnsw-default",
        "kind": "hnsw",
        "hnswParameters": {
          "m": 4,
          "efConstruction": 400,
          "efSearch": 500,
          "metric": "cosine"
        }
      }
    ],
    "profiles": [
      {
        "name": "default-profile",
        "algorithm": "hnsw-default"
      }
    ]
  }
}
```

## Troubleshooting

### Embedding Issues
- Check your Azure OpenAI API key and endpoint in the .env file
- Verify your embedding model deployment name matches "text-embedding-3-small"
- Ensure your API versions are correct in the .env file

### Azure Search Issues
- Confirm your index schema exactly matches the one in index_schema.json
- Make sure the "embedding_vector" field name is used consistently
- Verify your search endpoint and admin key
- Check that the vectorSearchProfile and algorithm configurations are correct
- Document IDs are automatically created using content hashes for uniqueness

### General Issues
- Look for detailed logs in the console output (INFO level)
- Use Python 3.8+ with the virtual environment
- Ensure all required packages are installed from requirements.txt
- For large files, consider reducing the chunk size with --max-tokens
- Metadata is properly stored as a JSON string in the index

## Advanced Features

### Content-Based Document IDs
The system uses SHA1 hashes of document content as IDs, which:
- Ensures valid characters for Azure Search keys
- Prevents duplicates with identical content
- Makes the system idempotent (safe to run multiple times)

### Automatic Schema Compliance
Documents are automatically formatted to comply with the Azure Search index schema, including:
- Field name standardization
- Metadata JSON serialization
- Vector field naming conventions

### Robust Error Handling
The system includes multiple fallback mechanisms:
- Multiple vector search configurations are attempted
- Keyword search fallback if vector search fails
- Comprehensive logging for debugging
