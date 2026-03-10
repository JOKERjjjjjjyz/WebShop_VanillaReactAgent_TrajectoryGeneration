#!/bin/bash
SESSION_NAME="webshop_rollout"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create a new detached session
tmux new-session -d -s $SESSION_NAME

# Pane 1 (vLLM Server)
tmux rename-window -t $SESSION_NAME:0 'Server_and_Workers'
tmux send-keys -t $SESSION_NAME 'HF_HOME=/data1/yanze/.cache/huggingface CUDA_VISIBLE_DEVICES=0 /data1/yanze/miniconda3/envs/agent/bin/vllm serve /data1/yanze/.cache/huggingface/hub/qwen3.5-9b --port 8000 --tensor-parallel-size 1 --max-model-len 50000 --language-model-only --gpu-memory-utilization 0.85 --served-model-name Qwen3.5-9B' C-m

# Split the window vertically for Pane 2
tmux split-window -v -t $SESSION_NAME

# Wait for vLLM to be ready
tmux send-keys -t $SESSION_NAME 'echo "Waiting for vLLM server to start on port 8000..."' C-m
tmux send-keys -t $SESSION_NAME 'while ! curl -s http://localhost:8000/health > /dev/null; do sleep 5; done' C-m
tmux send-keys -t $SESSION_NAME 'echo "vLLM is READY!"' C-m


# Pane 2 (Worker Pool) - 500 tasks, 4 workers, max_steps 1000
tmux send-keys -t $SESSION_NAME 'JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 WEBSHOP_PATH=/data1/yanze/WebShop PYTHONUNBUFFERED=1 /data1/yanze/miniconda3/envs/webshop/bin/python /data1/yanze/PARG_WebStore/scripts/run_baseline.py --workers 4 --max-tasks 500 --max-steps 1000 --llm-model Qwen3.5-9B' C-m

echo "Started tmux session: $SESSION_NAME"
echo "To view, run: tmux a -t $SESSION_NAME"
