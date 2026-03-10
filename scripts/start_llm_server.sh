#!/bin/bash
# ============================================================
# Start Qwen3.5-4B LLM Server (vLLM OpenAI-compatible API)
# ============================================================
# Uses vLLM 0.17.0 nightly with continuous batching.
#
# Usage:
#   ./start_llm_server.sh          # defaults: port 8000, gpu 0
#   ./start_llm_server.sh 8001 1   # port 8001, gpu 1
#
# After starting, test with:
#   curl http://localhost:8000/health
#   curl http://localhost:8000/v1/models
# ============================================================

PORT=${1:-8000}
CUDA_DEVICE=${2:-0}
MODEL_PATH=/data1/yanze/.cache/huggingface/hub/qwen3.5-4b
VLLM_BIN=/data1/yanze/miniconda3/envs/agent/bin/vllm

echo "[LLM Server] vLLM 0.17.0 + Qwen3.5-4B"
echo "[LLM Server] Port:   $PORT"
echo "[LLM Server] GPU:    cuda:$CUDA_DEVICE"
echo ""

export HF_HOME=/data1/yanze/.cache/huggingface
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

exec "$VLLM_BIN" serve "$MODEL_PATH" \
    --port "$PORT" \
    --served-model-name "Qwen3.5-4B" \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --language-model-only \
    --gpu-memory-utilization 0.85
