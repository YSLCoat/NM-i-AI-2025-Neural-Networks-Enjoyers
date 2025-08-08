# model.py

import pickle
import requests
import json
import re
from typing import Tuple
import numpy as np

import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

print("--- MODEL.PY: Initializing models ---")

# --- Configuration ---
# You can adjust these parameters as needed.
# NOTE: You will need to create and place the faiss_index.bin and clean_chunks.pkl files in the main directory.
LLM_MODEL = "gemma2:9b"
EMBEDDING_MODEL = 'BAAI/bge-large-en-v1.5'
RERANKER_MODEL = 'BAAI/bge-reranker-large'
FAISS_INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "clean_chunks.pkl"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
TOP_K_RETRIEVE = 50  # How many initial candidates to retrieve
TOP_N_RERANKED = 3   # How many of the best reranked candidates to use for context

RETRIEVER = None
CROSS_ENCODER = None

def load_models():
    """Loads all necessary models into memory."""
    global RETRIEVER, CROSS_ENCODER
    if RETRIEVER is None:
        try:
            print(f"Loading Bi-Encoder embedding model: {EMBEDDING_MODEL}")
            retriever_model = SentenceTransformer(EMBEDDING_MODEL, device='cuda')
            
            print(f"Loading FAISS index from: {FAISS_INDEX_PATH}")
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            
            print(f"Loading chunks from: {CHUNKS_PATH}")
            with open(CHUNKS_PATH, "rb") as f:
                chunks = pickle.load(f)
            
            RETRIEVER = {"model": retriever_model, "index": faiss_index, "chunks": chunks}
            print("Retriever initialized successfully.")

        except Exception as e:
            print(f"FATAL: Failed to initialize retriever: {e}")
            raise
            
    if CROSS_ENCODER is None:
        try:
            print(f"Loading Cross-Encoder reranker model: {RERANKER_MODEL}")
            CROSS_ENCODER = CrossEncoder(RERANKER_MODEL, device='cuda')
            print("Cross-Encoder initialized successfully.")
        except Exception as e:
            print(f"FATAL: Failed to initialize cross-encoder: {e}")
            raise

def predict(statement: str) -> Tuple[int, int]:
    """
    Predicts truth and topic for a statement using the full RAG pipeline.
    """
    if not RETRIEVER or not CROSS_ENCODER:
        print("Error: Models are not loaded. Cannot make a prediction.")
        return (0, 0) # Return a default/error value

    query_embedding = RETRIEVER["model"].encode(statement, normalize_embeddings=True).reshape(1, -1)
    _, indices = RETRIEVER["index"].search(query_embedding, TOP_K_RETRIEVE)
    retrieved_chunks = [RETRIEVER["chunks"][i] for i in indices[0]]

    chunk_texts = [f"{chunk['section_title']}: {chunk['content']}" for chunk in retrieved_chunks]
    pairs = [[statement, text] for text in chunk_texts]
    scores = CROSS_ENCODER.predict(pairs)
    sorted_indices = np.argsort(scores)[::-1]
    reranked_chunks = [retrieved_chunks[i] for i in sorted_indices]

    top_n_chunks = reranked_chunks[:TOP_N_RERANKED]

    if not top_n_chunks:
        print("Warning: No relevant chunks found after reranking.")
        return (0, 0)

    context_parts = [f"Section: {chunk['section_title']}\n\n{chunk['content']}" for chunk in top_n_chunks]
    context_text = "\n\n---\n\n".join(context_parts)
    topic_id = top_n_chunks[0]['topic_id']

    prompt = f"""Context from medical articles:\n---\n{context_text}\n---\n
        Statement to evaluate: "{statement}"

        Task:
        1.  **Analyze**: First, carefully read the statement and identify its key claims.
        2.  **Verify**: Second, go through the provided context section by section and check if it supports or refutes each key claim.
        3.  **Conclude**: Third, based on your verification, determine if the entire statement is true or false. The statement is only true if ALL its claims are supported by the context.
        4.  **Output**: Finally, respond with a single, raw JSON object with two keys: "statement_is_true" (1 for true, 0 for false) and "statement_topic" (the integer topic ID, which is {topic_id}).

        Do not add any explanation or markdown in your final output. Just the raw JSON.
        """

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.01}
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        response_text = response.json().get('response', '')
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        
        if json_match:
            prediction = json.loads(json_match.group(0))
            return (prediction.get("statement_is_true", 0), prediction.get("statement_topic", 0))
        else:
            print(f"Error: No JSON object found in Ollama response: {response_text}")
            return (0, 0)
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"Error during API call or JSON decoding: {e}")
        return (0, 0)

load_models()
print("--- MODEL.PY: All models loaded and ready. ---")