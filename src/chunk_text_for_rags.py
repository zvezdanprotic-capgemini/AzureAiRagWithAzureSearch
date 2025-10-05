#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chunk plain-text files into token-bounded windows with overlap for RAG.
- Primary boundef make_chunk_record(
    source_path: str,
    chunk_index: int,
    content: str
) -> Dict[str, Any]:
    # A stable ID ydef write_jsonl(records: Iterable[Dict[str, Any]], output_path: str) -> None:
    output_dir = os.path.dirname(os.path.abspath(output_path))
    logger.info(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Writing records to {output_path}...")
    record_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            json_line = json.dumps(r, ensure_ascii=False) + "\n"
            f.write(json_line)
            f.flush()  # Force write to disk immediately
            record_count += 1
            if record_count % 100 == 0:
                print(f"  ... wrote {record_count} chunks", file=sys.stdout, flush=True)
                logger.info(f"Wrote {record_count} records so far")
    
    print(f"  ... finished writing a total of {record_count} chunks.", file=sys.stdout, flush=True)
    logger.info(f"Completed writing {record_count} records to {output_path}") re-ingesting (helps upserts)
    doc_id = f"{os.path.basename(source_path)}::chunk-{chunk_index:05d}"
    logger.debug(f"Creating chunk record {doc_id} with {len(content)} chars of content")
    
    content_hash = sha1(content)
    logger.debug(f"Content hash for chunk {doc_id}: {content_hash}")
    
    return {
        "id": doc_id,
        "source_path": os.path.relpath(source_path),
        "chunk_index": chunk_index,
        "content": content,
        "content_sha1": content_hash,
        # room for future: metadata you may filter on in Azure AI Search
        "metadata": {
            "filename": os.path.basename(source_path),
            "ext": os.path.splitext(source_path)[1].lower()
        }
        # NOTE: you'll typically add 'embedding_vector' after you generate embeddings.
    }> sentences -> tokens (with overlap)
- Tokenizer: cl100k_base (compatible with OpenAI text-embedding-3-small)
- Output: JSONL; one chunk per line with metadata

Usage:
    python chunk_texts_for_rag.py --input ./data --output ./chunks.jsonl --max-tokens 512 --overlap 50
"""

import os
import re
import json
import argparse
import hashlib
import logging
import sys
import logging
import sys
import time
from typing import Iterable, List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Default level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('chunk_text_rag')

# ---- Tokenizer (tiktoken) -----------------------------------------------
# If tiktoken isn't installed, fall back to a naive whitespace tokenizer.
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    logger.info("Successfully loaded cl100k_base encoding from tiktoken")
    
    def encode_tokens(text: str) -> List[int]:
        tokens = _ENC.encode(text)
        logger.debug(f"Encoded {len(text)} chars into {len(tokens)} tokens")
        return tokens
        
    def decode_tokens(tokens: List[int]) -> str:
        text = _ENC.decode(tokens)
        logger.debug(f"Decoded {len(tokens)} tokens into {len(text)} chars")
        return text
except Exception as e:
    logger.warning(f"tiktoken not available ({str(e)}); falling back to naive tokenization.")
    print("[WARN] tiktoken not available; falling back to naive tokenization.")
    
    def encode_tokens(text: str) -> List[str]:
        tokens = text.split()
        logger.debug(f"Naive encoded {len(text)} chars into {len(tokens)} tokens")
        return tokens
        
    def decode_tokens(tokens: List[str]) -> str:
        text = " ".join(tokens)
        logger.debug(f"Naive decoded {len(tokens)} tokens into {len(text)} chars")
        return text


# ---- Utilities -----------------------------------------------------------
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

def iter_text_files(root: str, extensions=(".txt",)) -> Iterable[str]:
    logger.info(f"Searching for files with extensions {extensions} in {root}")
    file_count = 0
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(extensions):
                file_path = os.path.join(dirpath, f)
                logger.debug(f"Found file: {file_path}")
                file_count += 1
                yield file_path
    logger.info(f"Found {file_count} files with matching extensions")

def normalize_text(s: str) -> str:
    logger.debug(f"Normalizing text of length {len(s)}")
    # Collapse consecutive whitespace, keep newlines (for paragraph detection)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # compress spaces/tabs
    s = re.sub(r"[ \t]+", " ", s)
    # trim lines
    s = "\n".join(line.strip() for line in s.split("\n"))
    # collapse 3+ newlines down to 2 (paragraph marker)
    s = re.sub(r"\n{3,}", "\n\n", s)
    result = s.strip()
    logger.debug(f"Normalized text length: {len(result)} chars")
    return result

def split_paragraphs(text: str) -> List[str]:
    # Split by blank lines as paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    logger.debug(f"Split text into {len(paragraphs)} paragraphs")
    return paragraphs

def split_sentences(paragraph: str) -> List[str]:
    # Simple sentence splitter; adjust for your language/domain
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]
    logger.debug(f"Split paragraph of length {len(paragraph)} into {len(sentences)} sentences")
    return sentences

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ---- Chunking core -------------------------------------------------------
def chunk_by_tokens_from_segments(
    segments: List[str],
    max_tokens: int = 512,
    overlap: int = 50,
    max_chunks_limit: int = 10000  # Safety limit to prevent infinite loops
) -> Iterable[str]:
    """
    Accumulate segments (sentences or paragraphs) into token windows
    that respect max_tokens, with an overlap in tokens between chunks.
    """
    logger.info(f"Chunking {len(segments)} segments with max_tokens={max_tokens}, overlap={overlap}")
    if max_tokens <= 0:
        logger.error("max_tokens must be > 0")
        raise ValueError("max_tokens must be > 0")
    if overlap < 0 or overlap >= max_tokens:
        logger.error(f"Invalid overlap: {overlap} (must be >= 0 and < {max_tokens})")
        raise ValueError("overlap must be >= 0 and < max_tokens")

    current_tokens: List[int] = []
    chunks_generated = 0
    oversized_segments = 0
    
    # Calculate a reasonable maximum expected chunks based on input size
    total_tokens_estimate = sum(len(encode_tokens(seg)) for seg in segments)
    reasonable_chunks_estimate = (total_tokens_estimate // (max_tokens - overlap)) + len(segments) + 10
    chunks_limit = min(max_chunks_limit, max(100, reasonable_chunks_estimate * 3))
    logger.info(f"Total estimated tokens: {total_tokens_estimate}, expecting ~{reasonable_chunks_estimate} chunks (limit: {chunks_limit})")
    
    for seg_idx, seg in enumerate(segments):
        # Safety check to prevent infinite loops
        if chunks_generated >= chunks_limit:
            logger.error(f"Exceeded maximum chunk limit ({chunks_limit}). Stopping to prevent infinite loop.")
            raise RuntimeError(f"Exceeded maximum chunk limit ({chunks_limit}). Process aborted to prevent infinite loop.")
            
        seg_tokens = encode_tokens(seg)
        logger.debug(f"Segment {seg_idx}: {len(seg_tokens)} tokens")
        
        # If a single segment is too large, hard-split by tokens
        if len(seg_tokens) > max_tokens:
            logger.debug(f"Segment {seg_idx} exceeds max_tokens ({len(seg_tokens)} > {max_tokens}), hard-splitting")
            oversized_segments += 1
            start = 0
            while start < len(seg_tokens):
                end = min(start + max_tokens, len(seg_tokens))
                chunk = decode_tokens(seg_tokens[start:end])
                chunks_generated += 1
                logger.debug(f"Generated hard-split chunk {chunks_generated}: {len(chunk)} chars, {len(seg_tokens[start:end])} tokens")
                yield chunk
                
                if end == len(seg_tokens):
                    # Add overlap with next segments via the standard mechanism
                    current_tokens = seg_tokens[max(0, end - overlap):end]
                    logger.debug(f"Keeping {len(current_tokens)} tokens as overlap for next segments")
                    break  # Exit this loop to avoid processing this segment again
                else:
                    # Move start position forward, accounting for overlap
                    start = end - overlap
                    if start >= end:
                        logger.error(f"Infinite loop detected in hard-split: start={start}, end={end}")
                        break
            continue

        # Try to add the segment to the current window
        if len(current_tokens) + len(seg_tokens) <= max_tokens:
            logger.debug(f"Adding segment {seg_idx} ({len(seg_tokens)} tokens) to current window")
            current_tokens.extend(seg_tokens)
        else:
            # Emit current chunk
            if current_tokens:
                chunk = decode_tokens(current_tokens)
                chunks_generated += 1
                logger.debug(f"Generated chunk {chunks_generated}: {len(chunk)} chars from {len(current_tokens)} tokens")
                yield chunk
                
            # Start a new window with overlap from previous
            if overlap > 0 and current_tokens:
                current_tokens = current_tokens[-overlap:]
                logger.debug(f"Keeping {len(current_tokens)} tokens as overlap for next window")
            else:
                current_tokens = []
                
            # Add new segment (it must fit because we handled >max above)
            logger.debug(f"Starting new window with segment {seg_idx} ({len(seg_tokens)} tokens)")
            current_tokens.extend(seg_tokens)

    # Emit the last window
    if current_tokens:
        chunk = decode_tokens(current_tokens)
        chunks_generated += 1
        logger.debug(f"Generated final chunk {chunks_generated}: {len(chunk)} chars from {len(current_tokens)} tokens")
        yield chunk
        
    logger.info(f"Total chunks generated: {chunks_generated} (including {oversized_segments} oversized segments)")

def chunk_text(
    text: str,
    max_tokens: int = 512,
    overlap: int = 50
) -> Iterable[str]:
    """
    Hybrid approach:
      - split into paragraphs
      - split each paragraph into sentences
      - pack sentences into token windows with overlap
    This is a generator function that yields chunks directly to avoid memory accumulation.
    """
    logger.info(f"Chunking text of length {len(text)} chars with max_tokens={max_tokens}, overlap={overlap}")
    
    # Sanity check: if input is too small, don't bother with complex chunking
    if len(text) < 100:
        logger.info(f"Text is very short ({len(text)} chars), yielding as single chunk")
        yield text
        return
    
    normalized_text = normalize_text(text)
    paragraphs = split_paragraphs(normalized_text)
    logger.info(f"Split text into {len(paragraphs)} paragraphs")
    
    # If no paragraphs were found (e.g., just whitespace), return early
    if not paragraphs:
        logger.warning("No paragraphs found after normalization")
        return
    
    # Process paragraphs into sentences
    sentences: List[str] = []
    for p_idx, p in enumerate(paragraphs):
        p_sentences = split_sentences(p)
        logger.debug(f"Paragraph {p_idx}: Split into {len(p_sentences)} sentences")
        # If no sentences were found, use the paragraph as a single sentence
        if not p_sentences:
            sentences.append(p)
        else:
            sentences.extend(p_sentences)
    
    if not sentences:
        logger.warning("No sentences found after paragraph splitting")
        # If no sentences, but we had paragraphs, use the paragraphs
        sentences = paragraphs
    
    logger.info(f"Total sentences: {len(sentences)}")
    
    # Sanity check against empty segments
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        logger.warning("No non-empty sentences found")
        yield normalized_text
        return
    
    # Pass the sentences to the chunker and yield each chunk directly
    chunks_generated = 0
    try:
        for chunk in chunk_by_tokens_from_segments(sentences, max_tokens, overlap):
            chunks_generated += 1
            if len(chunk.strip()) > 0:  # Skip empty chunks
                logger.debug(f"Yielding chunk {chunks_generated} of {len(chunk)} chars")
                yield chunk
            else:
                logger.warning(f"Skipping empty chunk {chunks_generated}")
        
        logger.info(f"Created {chunks_generated} chunks from text")
    except RuntimeError as e:
        logger.error(f"Error during chunking: {str(e)}")
        # If chunking fails but we have some chunks already, let those through
        if chunks_generated == 0:
            # As a fallback, create simple chunks based on character count
            logger.info("Falling back to simple character-based chunking")
            simple_chunk_size = max(100, max_tokens * 4)  # Rough estimate of chars per token
            for i in range(0, len(normalized_text), simple_chunk_size):
                yield normalized_text[i:i+simple_chunk_size]
                chunks_generated += 1
            logger.info(f"Created {chunks_generated} fallback chunks from text")

# ---- JSONL writer --------------------------------------------------------
def make_chunk_record(
    source_path: str,
    chunk_index: int,
    content: str
) -> Dict[str, Any]:
    # A stable ID you can reuse when re-ingesting (helps upserts)
    doc_id = f"{os.path.basename(source_path)}::chunk-{chunk_index:05d}"
    return {
        "id": doc_id,
        "source_path": os.path.relpath(source_path),
        "chunk_index": chunk_index,
        "content": content,
        "content_sha1": sha1(content),
        # room for future: metadata you may filter on in Azure AI Search
        "metadata": {
            "filename": os.path.basename(source_path),
            "ext": os.path.splitext(source_path)[1].lower()
        }
        # NOTE: you’ll typically add 'embedding_vector' after you generate embeddings.
    }

def write_jsonl(records: Iterable[Dict[str, Any]], output_path: str) -> None:
    output_dir = os.path.dirname(os.path.abspath(output_path))
    logger.info(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Writing records to {output_path}...")
    record_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            record_count += 1
            if record_count % 5 == 0:
                print(f"  ... wrote {record_count} chunks", file=sys.stdout, flush=True)
                logger.info(f"Wrote {record_count} records so far")
    
    print(f"  ... finished writing a total of {record_count} chunks.", file=sys.stdout, flush=True)
    logger.info(f"Completed writing {record_count} records to {output_path}")

def process_files_and_generate_records(paths_to_process: Iterable[str], max_tokens: int, overlap: int) -> Iterable[Dict[str, Any]]:
    """Processes each file and yields chunk records one by one."""
    file_count = 0
    total_chunks = 0
    for path in paths_to_process:
        file_count += 1
        logger.info(f"Processing file {file_count}: {path}")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            logger.info(f"Read {len(raw)} characters from {path}")
            
            # Process chunks one at a time using a streaming approach
            chunk_count = 0
            for c in chunk_text(raw, max_tokens=max_tokens, overlap=overlap):
                record = make_chunk_record(path, chunk_count, c)
                chunk_count += 1
                if chunk_count % 100 == 0:
                    logger.info(f"Processed {chunk_count} chunks from {path} so far")
                yield record
            
            logger.info(f"Yielded {chunk_count} chunks from {path}")
            total_chunks += chunk_count
            
        except Exception as e:
            logger.error(f"Error processing file {path}: {str(e)}")
    logger.info(f"Finished processing {file_count} files, yielding a total of {total_chunks} chunks.")

# ---- CLI -----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Chunk .txt files for RAG.")
    ap.add_argument("--input", required=True, help="Folder or file with .txt files")
    ap.add_argument("--output", required=True, help="Output JSONL file")
    ap.add_argument("--max-tokens", type=int, default=512, help="Max tokens per chunk")
    ap.add_argument("--overlap", type=int, default=50, help="Token overlap between chunks")
    ap.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = ap.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Resolve input path to be absolute
    input_path = os.path.abspath(args.input)
    logger.info(f"Resolved input path: {input_path}")

    # Check if the input path exists
    if not os.path.exists(input_path):
        logger.error(f"Input path does not exist: {input_path}")
        logger.error(f"Current working directory is: {os.getcwd()}")
        logger.error("Please provide a valid path to a file or directory.")
        sys.exit(1)
    
    logger.info(f"Starting chunking process with parameters: input={input_path}, output={args.output}, max-tokens={args.max_tokens}, overlap={args.overlap}")
    
    start_time = time.time()
    
    # Determine if input is a file or directory
    if os.path.isdir(input_path):
        paths_to_process = iter_text_files(input_path, extensions=(".txt",))
    elif os.path.isfile(input_path) and input_path.lower().endswith(".txt"):
        paths_to_process = [input_path]
    else:
        logger.error(f"Input path is not a valid directory or .txt file: {input_path}")
        sys.exit(1)

    # Create a generator for the records
    record_generator = process_files_and_generate_records(paths_to_process, args.max_tokens, args.overlap)
    
    # Write the records to the output file as they are generated
    write_jsonl(record_generator, args.output)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
    # The final count is now logged inside write_jsonl, so we can't get it here easily.
    # The high-level message is sufficient.
    print(f"[OK] Chunking complete. Output written to {args.output}")

if __name__ == "__main__":
    try:
        logger.info("Starting chunk_text_for_rags.py")
        main()
        logger.info("Finished successfully")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)
