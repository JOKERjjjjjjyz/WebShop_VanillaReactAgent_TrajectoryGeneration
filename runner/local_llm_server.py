#!/usr/bin/env python3
"""
Local LLM Server for PARG_WebStore
====================================
Runs Qwen3.5-4B (or any HuggingFace causal LM) as an OpenAI-compatible
HTTP server using the Transformers library + Flask.

Endpoints (OpenAI-compatible):
    POST /v1/chat/completions
    GET  /v1/models

Usage:
    # Start server (in agent env with CUDA):
    CUDA_VISIBLE_DEVICES=0 \\
    /data1/yanze/miniconda3/envs/agent/bin/python \\
    /data1/yanze/PARG_WebStore/runner/local_llm_server.py \\
    --model /data1/yanze/.cache/huggingface/hub/qwen3.5-4b \\
    --port 8000

    # Then use from any env with the standard OpenAI client:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
    resp = client.chat.completions.create(
        model="Qwen3.5-4B",
        messages=[{"role": "user", "content": "Hello!"}]
    )
"""

import argparse
import time
import uuid
import json
import threading
import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

app = Flask(__name__)

# Global model/tokenizer (loaded at startup)
_model = None
_tokenizer = None
_model_name = "Qwen3.5-4B"
_lock = threading.Lock()  # serialize generation requests


def load_model(model_path: str, device: str = "auto", dtype=torch.bfloat16):
    global _model, _tokenizer
    print(f"[LLM Server] Loading tokenizer from {model_path}...")
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"[LLM Server] Loading model...")
    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    _model.eval()
    print(f"[LLM Server] Model loaded! Device map: {_model.hf_device_map if hasattr(_model, 'hf_device_map') else device}")


def generate(messages: list, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
    """Run generation for a chat messages list."""
    # Apply chat template
    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        # Qwen3.5 supports thinking mode; disable for speed in ReAct usage
        enable_thinking=False,
    )
    inputs = _tokenizer([text], return_tensors="pt").to(_model.device)

    with torch.no_grad():
        generated_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=(temperature > 0),
            pad_token_id=_tokenizer.eos_token_id,
        )

    # Slice off the prompt tokens
    new_ids = generated_ids[0][inputs.input_ids.shape[1]:]
    response = _tokenizer.decode(new_ids, skip_special_tokens=True)
    return response.strip()


@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list",
        "data": [{
            "id": _model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }]
    })


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json(force=True)
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.7)

    with _lock:
        content = generate(messages, max_new_tokens=max_tokens, temperature=temperature)

    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": -1,
            "total_tokens": -1,
        }
    }
    return jsonify(response)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": _model_name})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="/data1/yanze/.cache/huggingface/hub/qwen3.5-4b",
                        help="Path to HuggingFace model directory")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--model-name", type=str, default="Qwen3.5-4B",
                        help="Name to expose via /v1/models")
    args = parser.parse_args()

    _model_name = args.model_name
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    load_model(args.model, dtype=dtype_map[args.dtype])

    print(f"[LLM Server] Starting on http://{args.host}:{args.port}")
    print(f"[LLM Server] Model: {args.model_name}")
    print(f"[LLM Server] Test: curl http://localhost:{args.port}/health")
    app.run(host=args.host, port=args.port, debug=False, threaded=False)
