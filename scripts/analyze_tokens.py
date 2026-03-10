
import json
import os
import sys
from transformers import AutoTokenizer

# Path configuration
MODEL_PATH = "/data1/yanze/.cache/huggingface/hub/qwen3.5-4b"
LOG_FILE = "/data1/yanze/PARG_WebStore/logs/raw/webshop/react_v1/2026-03-06_baseline.jsonl"

def analyze_tokens():
    if not os.path.exists(LOG_FILE):
        print(f"Error: Log file not found at {LOG_FILE}")
        return

    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    total_unique_tokens_list = []
    final_context_tokens_list = []
    
    print(f"Analyzing {LOG_FILE}...")
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            trajectory = data.get('trajectory', [])
            
            # 1. Total Unique Tokens (sum of all text appearing in the trajectory nodes)
            # This represents the "raw data size" of the episode.
            episode_text = ""
            for node in trajectory:
                payload = node.get('payload', {})
                # Extract text from whatever payload keys exist
                for key in ['raw_text', 'action_str']:
                    if key in payload:
                        episode_text += str(payload[key]) + " "
            
            unique_tokens = len(tokenizer.encode(episode_text))
            total_unique_tokens_list.append(unique_tokens)
            
            # 2. Final Context Tokens (cumulative size at the last step)
            # This is what the LLM actually sees at step N.
            # In ReAct: Prompt + Obs1 + (T1 + A1 + Obs2) + (T2 + A2 + Obs3) ...
            # We can approximate this by summing thoughts/actions/obs in sequence order.
            context_text = "" # Simplified prompt accumulation
            for node in trajectory:
                payload = node.get('payload', {})
                for key in ['raw_text', 'action_str']:
                    if key in payload:
                        context_text += str(payload[key]) + " "
            
            final_context_tokens = len(tokenizer.encode(context_text))
            final_context_tokens_list.append(final_context_tokens)

    if not total_unique_tokens_list:
        print("No episodes found in log.")
        return

    def stats(arr):
        return {
            "min": min(arr),
            "max": max(arr),
            "avg": sum(arr) / len(arr)
        }

    unique_stats = stats(total_unique_tokens_list)
    context_stats = stats(final_context_tokens_list)

    print("\n" + "="*40)
    print("  QWEN3.5 TOKEN ANALYSIS REPORT")
    print("="*40)
    print(f"Analyzed {len(total_unique_tokens_list)} episodes.")
    print("-" * 40)
    print("1. Episode 'Raw' Token Count (Sum of all obs + thoughts + actions):")
    print(f"   - Average: {unique_stats['avg']:7.1f} tokens")
    print(f"   - Minimum: {unique_stats['min']:7d} tokens")
    print(f"   - Maximum: {unique_stats['max']:7d} tokens")
    print("-" * 40)
    print("2. Estimated Final Context Window Size (Last step context):")
    print(f"   - Average: {context_stats['avg']:7.1f} tokens")
    print(f"   - Minimum: {context_stats['min']:7d} tokens")
    print(f"   - Maximum: {context_stats['max']:7d} tokens")
    print("="*40)

if __name__ == "__main__":
    analyze_tokens()
