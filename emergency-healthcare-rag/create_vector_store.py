import os
import argparse
import pickle
import time
import faiss
from sentence_transformers import SentenceTransformer

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