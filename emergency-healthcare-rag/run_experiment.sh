#!/bin/bash
set -e

EMBEDDING_MODEL="BAAI/bge-large-en-v1.5"
LLM_MODEL="gemma:7b" 

CHUNKS_FILENAME="clean_chunks.pkl"
FAISS_INDEX_FILENAME="faiss_index.bin"

# Define the paths to the Python scripts and data directories
CREATE_STORE_SCRIPT="create_vector_store.py"
INFERENCE_SCRIPT="inference.py"
STATEMENTS_DIR="/home/torf/NM-i-AI-2025-Neural-Networks-Enjoyers/emergency-healthcare-rag/data/train/statements/"
GROUND_TRUTH_DIR="/home/torf/NM-i-AI-2025-Neural-Networks-Enjoyers/emergency-healthcare-rag/data/train/answers/"


echo "--- Step 1: Creating new vector store with embedding model: $EMBEDDING_MODEL ---"
python "$CREATE_STORE_SCRIPT" \
    --model "$EMBEDDING_MODEL" \
    --chunks_filename "$CHUNKS_FILENAME" \
    --faiss_index_filename "$FAISS_INDEX_FILENAME"

echo "--- Vector store created successfully. ---"
echo ""


echo "--- Step 2: Running evaluation with LLM: $LLM_MODEL ---"
python "$INFERENCE_SCRIPT" \
    --llm_model "$LLM_MODEL" \
    --embedding_model "$EMBEDDING_MODEL" \
    --faiss_index "$FAISS_INDEX_FILENAME" \
    --chunks_file "$CHUNKS_FILENAME" \
    --statements_dir "$STATEMENTS_DIR" \
    --ground_truth_dir "$GROUND_TRUTH_DIR"

echo "--- Experiment finished successfully! ---"