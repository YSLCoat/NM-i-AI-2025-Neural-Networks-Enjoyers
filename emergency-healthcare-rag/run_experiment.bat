@echo off
setlocal

set "PROJECT_BASE_PATH=%USERPROFILE%\NM-i-AI-2025-Neural-Networks-Enjoyers\emergency-healthcare-rag"
set "STATEMENTS_DIR=%PROJECT_BASE_PATH%\data\train\statements"
set "GROUND_TRUTH_DIR=%PROJECT_BASE_PATH%\data\train\answers"


REM --- Model and File Configuration ---
set "EMBEDDING_MODEL=BAAI/bge-large-en-v1.5"
REM --- NEW: Define the Cross-Encoder model for reranking ---
set "RERANKER_MODEL=BAAI/bge-reranker-large"
set "LLM_MODEL=gemma2:9b"

REM Giving the powerful reranker more candidates can help. 10 is a good starting point.
set "TOP_K_FOR_RERANKING=10"
set "CHUNKS_FILENAME=clean_chunks.pkl"
set "FAISS_INDEX_FILENAME=faiss_index.bin"

REM Define the paths to the Python scripts
set "CREATE_STORE_SCRIPT=create_vector_store.py"
set "INFERENCE_SCRIPT=inference.py"


echo --- Step 1: Creating new vector store with embedding model: %EMBEDDING_MODEL% ---
@REM python "%CREATE_STORE_SCRIPT%" --model "%EMBEDDING_MODEL%" --chunks_filename "%CHUNKS_FILENAME%" --faiss_index_filename "%FAISS_INDEX_FILENAME%"

echo --- Vector store created successfully. ---
echo.


echo --- Step 2: Running evaluation with LLM: %LLM_MODEL% ---
python "%INFERENCE_SCRIPT%" --llm_model "%LLM_MODEL%" --embedding_model "%EMBEDDING_MODEL%" --reranker_model "%RERANKER_MODEL%" --faiss_index "%FAISS_INDEX_FILENAME%" --chunks_file "%CHUNKS_FILENAME%" --statements_dir "%STATEMENTS_DIR%" --ground_truth_dir "%GROUND_TRUTH_DIR%" --top_k "%TOP_K_FOR_RERANKING%"

echo --- Experiment finished successfully! ---
echo.
pause