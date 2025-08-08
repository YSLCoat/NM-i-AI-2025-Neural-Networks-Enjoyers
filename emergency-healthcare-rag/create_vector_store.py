import os
import argparse
import pickle
import time
import faiss
from sentence_transformers import SentenceTransformer

import json
import re
import pathlib
import pickle
import logging
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_topics_index_mapping(path: str) -> dict: 
    with open(path) as f:
        mapping = json.load(f)
    return mapping

def create_clean_chunks(topic_id: int, raw_text: str, tokenizer_model: str = "BAAI/bge-m3") -> List[Dict[str, Any]]:
    unwanted_sections = {
        'Authors', 'Affiliations', 'Continuing Education Activity', 'Review Questions',
        'References', 'Disclosure', 'Comment on this article.',
        'Access free multiple choice questions on this topic.'
    }
    
    # Remove frontmatter
    text = re.sub(r'---\n(.*?)\n---', '', raw_text, flags=re.DOTALL)
    # Remove references section and citation marks like [1], [2], etc.
    text = re.sub(r'## References.*', '', text, flags=re.DOTALL)
    text = re.sub(r'\[\d+\]', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Clean up formatting
    text = text.replace('\\\'', '\'')
    text = re.sub(r'_(.*?)_', r'\1', text) # remove underscores that are not part of a word
    text = text.split('* \n\n  * Click here for a simplified version.')[0]
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    title = raw_text.split('\n')[0].lstrip('# ').strip()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,      # The target size for each chunk in tokens
        chunk_overlap=25,   # The number of tokens to overlap between chunks
        length_function=lambda x: len(tokenizer.encode(x)), # Use the tokenizer for length calculation
        separators=["\n\n## ", "\n\n", "\n", ". ", " ", ""], # How to split recursively
        add_start_index=False,
    )

    # We still use the '## ' as a primary separator to maintain logical structure.
    parts = re.split(r'(?=## )', text)
    chunks = []
    
    # Process the preamble (text before the first ## section)
    preamble_content = parts[0].strip()
    if preamble_content:
        # Prepend the main document title to the preamble
        preamble_with_title = f"# {title}\n\n{preamble_content}"
        sub_chunks = text_splitter.split_text(preamble_with_title)
        for sub_chunk in sub_chunks:
            chunks.append({
                'topic_id': topic_id,
                'section_title': title, # Use the main title for the preamble
                'content': sub_chunk
            })

    # Process the rest of the sections
    for part in parts[1:]:
        try:
            # Extract section title
            title_end_index = part.find('\n')
            section_title = part[2:title_end_index].strip() # remove '##'
            section_content = part[title_end_index:].strip()

            if section_title and section_content and section_title not in unwanted_sections:
                # ENHANCEMENT: Add main document title AND section title
                content_with_full_title = f"# {title}\n\n## {section_title}\n\n{section_content}"
                sub_chunks = text_splitter.split_text(content_with_full_title)
                
                for sub_chunk in sub_chunks:
                    chunks.append({
                        'topic_id': topic_id,
                        'section_title': section_title,
                        'content': sub_chunk,
                        'main_doc_title': title # Optional: store for clarity
                    })
        except (IndexError, ValueError):
            logging.warning(f"Could not process a malformed section part in topic {topic_id}")
            continue
            
    return chunks


def prepare_and_save_chunks(topics_dir: pathlib.Path, topics_json_path: pathlib.Path, output_file: pathlib.Path) -> List[Dict[str, Any]]:
    name_to_id_map = load_topics_index_mapping(topics_json_path)
    master_chunks_list = []
    
    logging.info(f"Starting to process markdown files in '{topics_dir}'...")
    
    topic_paths = [p for p in pathlib.Path(topics_dir).iterdir() if p.is_dir()]
    
    for topic_path in topic_paths:
        topic_name = topic_path.name
        topic_id = name_to_id_map.get(topic_name)
        
        if topic_id is None:
            logging.warning(f"No topic ID found for topic '{topic_name}'. Skipping.")
            continue
        
        logging.info(f"Processing topic: '{topic_name}' (ID: {topic_id})")
        
        content_parts = []
        for md_file_path in sorted(list(topic_path.glob('*.md'))):
            with open(md_file_path, 'r', encoding='utf-8') as f:
                content_parts.append(f.read())
        
        raw_text = "\n\n---\n\n".join(content_parts)

        if raw_text:
            chunks = create_clean_chunks(topic_id, raw_text)
            master_chunks_list.extend(chunks)
            logging.info(f"Generated {len(chunks)} chunks for topic '{topic_name}'.")

    logging.info(f"Total chunks generated: {len(master_chunks_list)}")
    logging.info(f"Saving the list to '{output_file}'")
    with open(output_file, "wb") as f:
        pickle.dump(master_chunks_list, f)
    
    return master_chunks_list

from data_utils import (
    prepare_and_save_chunks
)

def build_vector_store(model, chunk_list_filename, faiss_filename):
    try:
        with open(chunk_list_filename, "rb") as f:
            clean_chunks = pickle.load(f)
    except FileNotFoundError:
        return

    print(f"Loaded {len(clean_chunks)}")

    print("Loading sentence transformer model...")
    model = SentenceTransformer(model, device='cuda')

    texts_to_embed = [
        f"{chunk['section_title']}: {chunk['content']}" for chunk in clean_chunks
    ]

    print(f"\nGenerating embeddings for {len(texts_to_embed)} chunks...")
    start_time = time.time()
    embeddings = model.encode(
        texts_to_embed,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    end_time = time.time()
    print(f"Embeddings generated in {end_time - start_time:.2f} seconds.")

    embedding_dimension = embeddings.shape[1]
    print(f"Embedding dimension: {embedding_dimension}")

    index = faiss.IndexFlatIP(embedding_dimension)

    index.add(embeddings)
    
    print(f"FAISS index built. Total vectors in index: {index.ntotal}")

    faiss.write_index(index, faiss_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create vector store for rag pipeline')
    parser.add_argument('--data_dir', default=r"C:\Users\torfor\NM-i-AI-2025-Neural-Networks-Enjoyers\emergency-healthcare-rag\data")
    parser.add_argument('--chunks_filename')
    parser.add_argument('--model')
    parser.add_argument('--faiss_index_filename')
    args = parser.parse_args()

    master_chunk_list = prepare_and_save_chunks(os.path.join(args.data_dir, "topics"), os.path.join(args.data_dir, "topics.json"), args.chunks_filename)

    build_vector_store(args.model, args.chunks_filename, args.faiss_index_filename)