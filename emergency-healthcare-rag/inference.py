# inference.py

import argparse
import requests
import json
import re
import pathlib
import warnings
from tqdm import tqdm
import pickle

# --- Required Imports for GPU execution ---
import torch
import numpy as np
# MODIFIED: Import CrossEncoder
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # For quantization

# It's good practice to handle potential import errors for optional libraries
try:
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    print("Successfully initialized NVML for VRAM monitoring.")
except Exception as e:
    print(f"Warning: Could not initialize NVML for VRAM monitoring. Error: {e}")
    handle = None

warnings.filterwarnings("ignore", category=UserWarning)


class RAGRetriever:
    def __init__(self, index_path, chunks_path, embedding_model):
        print("Initializing the RAG Retriever...")
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # This is the BI-ENCODER for initial, fast retrieval
        self.model = SentenceTransformer(embedding_model, device=device)
        print(f"Retriever initialized successfully on device: {device}")

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        query_embedding = query_embedding.reshape(1, -1)
        _, indices = self.index.search(query_embedding, k)
        return [self.chunks[i] for i in indices[0]]

# --- NEW: Reranking function using a Cross-Encoder ---
def rerank_with_cross_encoder(cross_encoder_model: CrossEncoder, query: str, chunks: list[dict]) -> dict:
    """
    Reranks a list of chunks using a more powerful Cross-Encoder model.
    """
    # Create pairs of [query, chunk_text] for the cross-encoder
    chunk_texts = [f"{chunk['section_title']}: {chunk['content']}" for chunk in chunks]
    pairs = [[query, text] for text in chunk_texts]

    # The cross-encoder predicts a relevance score for each pair
    scores = cross_encoder_model.predict(pairs)

    # Get the index of the highest scoring chunk
    best_chunk_index = np.argmax(scores)

    # Return the single best chunk
    return chunks[best_chunk_index]


class RAGGenerator:
    def __init__(self, model_name: str, ollama_base_url: str = "http://localhost:11434"):
        print(f"Initializing RAG Generator to use Ollama model: '{model_name}'")
        self.model_name = model_name
        self.ollama_api_url = f"{ollama_base_url}/api/generate"
        try:
            requests.get(ollama_base_url)
            print(f"Successfully connected to Ollama at {ollama_base_url}")
        except requests.exceptions.ConnectionError:
            print(f"FATAL: Could not connect to Ollama at {ollama_base_url}. Please ensure Ollama is running.")
            exit()


    def generate(self, statement: str, context_chunk: dict) -> dict:
        context_text = f"Section: {context_chunk['section_title']}\n\n{context_chunk['content']}"
        topic_id = context_chunk['topic_id']

        prompt = f"""Context:\n---\n{context_text}\n---\nStatement: "{statement}"\n\nTask: Based ONLY on the provided context, respond with a single, raw JSON object with two keys: "statement_is_true" (1 for true, 0 for false) and "statement_topic" (the integer topic ID, which is {topic_id}). Do not add any explanation or markdown."""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.01
            }
        }

        try:
            response = requests.post(self.ollama_api_url, json=payload)
            response.raise_for_status() 

            response_text = response.json().get('response', '')
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group(0))
            else:
                print(f"\nError: No JSON object found in Ollama response: {response_text}")
                return {"statement_is_true": -1, "statement_topic": -1}

        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"\nError during API call or JSON decoding: {e}")
            return {"statement_is_true": -1, "statement_topic": -1}


def get_vram_usage_gb(device_handle):
    if not device_handle: return None
    info = pynvml.nvmlDeviceGetMemoryInfo(device_handle)
    return info.used / (1024**3)

def run_evaluation(args):
    print(f"--- Starting Evaluation for LLM: {args.llm_model} ---")
    
    retriever = RAGRetriever(args.faiss_index, args.chunks_file, args.embedding_model)
    generator = RAGGenerator(args.llm_model)
    # --- NEW: Load the Cross-Encoder model for reranking ---
    print(f"Loading Reranker Model: {args.reranker_model}")
    cross_encoder = CrossEncoder(args.reranker_model, device='cuda')


    statements_path = pathlib.Path(args.statements_dir)
    statement_files = sorted(list(statements_path.glob("*.txt")))
    
    total_statements = len(statement_files)
    correct_truth, correct_topic = 0, 0
    
    # Note: Cross-encoder reranking is more computationally intensive, so the process may be slower.
    for i, statement_file in enumerate(tqdm(statement_files, desc=f"Evaluating with Cross-Encoder")):
        with open(statement_file, 'r', encoding='utf-8') as f:
            statement_text = f.read().strip()

        statement_id = statement_file.stem
        ground_truth_file = pathlib.Path(args.ground_truth_dir) / f"{statement_id}.json"
        
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)

        # --- MODIFIED: RAG pipeline with Cross-Encoder Reranking ---
        # 1. Retrieve top-k chunks with the fast bi-encoder
        retrieved_chunks = retriever.retrieve(statement_text, k=args.top_k)
        
        # 2. Rerank these candidates with the more powerful cross-encoder
        best_chunk = rerank_with_cross_encoder(cross_encoder, statement_text, retrieved_chunks)
        
        # 3. Generate a response using only the best, reranked chunk
        prediction = generator.generate(statement_text, best_chunk)
        # --- End of Modification ---

        if prediction.get("statement_is_true") == ground_truth.get("statement_is_true"): correct_truth += 1
        if prediction.get("statement_topic") == ground_truth.get("statement_topic"): correct_topic += 1
            
        if (i + 1) % args.vram_check_interval == 0 and handle:
            vram = get_vram_usage_gb(handle)
            print(f" | VRAM after statement {i+1}: {vram:.2f} GB")
            
    truth_accuracy = (correct_truth / total_statements) * 100 if total_statements > 0 else 0
    topic_accuracy = (correct_topic / total_statements) * 100 if total_statements > 0 else 0

    print("\n--- Evaluation Complete ---")
    print(f"LLM Tested: {args.llm_model}")
    print(f"Retrieval Strategy: Top {args.top_k} retrieval with Cross-Encoder ({args.reranker_model}) Reranking")
    print(f"Total Statements: {total_statements}")
    print("\n--- Accuracy Scores ---")
    print(f"Statement Truth Accuracy: {truth_accuracy:.2f}%")
    print(f"Statement Topic Accuracy: {topic_accuracy:.2f}%")
    print("-----------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a full evaluation of the RAG pipeline using Hugging Face Transformers on GPU.")
    
    parser.add_argument("--llm_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Name of the Hugging Face model to use for generation.")
    parser.add_argument("--embedding_model", type=str, help="Name of the SentenceTransformer BI-ENCODER model for retrieval.")
    # NEW: Argument for the reranker model
    parser.add_argument("--reranker_model", type=str, default="BAAI/bge-reranker-large", help="Name of the SentenceTransformer CROSS-ENCODER model for reranking.")
    parser.add_argument("--statements_dir", type=str, default="/home/torf/NM-i-AI-2025-Neural-Networks-Enjoyers/emergency-healthcare-rag/data/train/statements/", help="Directory for statement .txt files.")
    parser.add_argument("--ground_truth_dir", type=str, default="/home/torf/NM-i-AI-2025-Neural-Networks-Enjoyers/emergency-healthcare-rag/data/train/answers/", help="Directory for ground truth .json files.")
    parser.add_argument("--faiss_index", type=str, default="faiss_index.bin", help="Path to the FAISS index file.")
    parser.add_argument("--chunks_file", type=str, default="clean_chunks.pkl", help="Path to the clean_chunks.pkl file.")
    parser.add_argument("--vram_check_interval", type=int, default=20, help="How often to check and print VRAM usage.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of chunks to retrieve from FAISS for reranking. More is better for a cross-encoder, but slower.")

    args = parser.parse_args()
    run_evaluation(args)