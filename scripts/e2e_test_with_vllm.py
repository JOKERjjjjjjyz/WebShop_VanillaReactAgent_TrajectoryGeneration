"""
End-to-end integration test: AgentInstance + vLLM + WebShop.

Simulates what MultiTurnAgentExecutor does:
1. AgentInstance.reset() → initial observation
2. vLLM generates action text
3. AgentInstance.step(action_text) → environment feedback
4. Repeat until done or max steps

Prerequisites:
    - WebShop pool server running: bash scripts/start_webshop_pool_server.sh 6001
    - vLLM running: vllm serve ... --port 8000

Usage:
    python scripts/e2e_test_with_vllm.py \
        --webshop-url http://localhost:6001 \
        --vllm-url http://localhost:8000 \
        --task-idx 0 \
        --max-steps 5
"""

import argparse
import asyncio
import os
import sys
import types
import importlib.util

import aiohttp
import requests


def load_agent_instance():
    """Load AgentInstance without full openrlhf import."""
    agent_mod = types.ModuleType("openrlhf")
    utils_mod = types.ModuleType("openrlhf.utils")
    agent_base_mod = types.ModuleType("openrlhf.utils.agent")

    class _AgentInstanceBase:
        pass

    class _MultiTurnAgentExecutor:
        def __init__(self, cls):
            self.cls = cls

    agent_base_mod.AgentInstanceBase = _AgentInstanceBase
    agent_base_mod.MultiTurnAgentExecutor = _MultiTurnAgentExecutor
    utils_mod.agent = agent_base_mod
    agent_mod.utils = utils_mod
    sys.modules["openrlhf"] = agent_mod
    sys.modules["openrlhf.utils"] = utils_mod
    sys.modules["openrlhf.utils.agent"] = agent_base_mod

    spec = importlib.util.spec_from_file_location(
        "agent_func_webshop",
        "/data1/yanze/OpenRLHF/examples/python/agent_func_webshop.py",
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.AgentInstance


def call_vllm(vllm_url, prompt_text, max_tokens=256):
    """Call vLLM OpenAI-compatible API to generate text continuation."""
    resp = requests.post(
        f"{vllm_url}/v1/completions",
        json={
            "model": "Qwen3.5-9B",
            "prompt": prompt_text,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stop": ["<|im_end|>", "<|im_start|>"],
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"]


async def run_episode(vllm_url, task_idx, max_steps):
    AgentInstance = load_agent_instance()

    # Create instance
    instance = AgentInstance.__new__(AgentInstance)
    instance.step_idx = 0
    instance.max_steps = max_steps
    instance.session_id = None
    instance._http_session = None

    # Reset
    states = {"observation": "placeholder", "label": str(task_idx)}
    result = await instance.reset(states)
    observation_text = result["observation"]

    print("=" * 70)
    print("INITIAL OBSERVATION (formatted prompt)")
    print("=" * 70)
    print(observation_text[:500])
    print("..." if len(observation_text) > 500 else "")

    total_reward = 0.0

    for step in range(max_steps):
        print(f"\n{'='*70}")
        print(f"STEP {step + 1}")
        print(f"{'='*70}")

        # Call vLLM to generate action
        action_text = call_vllm(vllm_url, observation_text, max_tokens=256)
        print(f"\n[LLM Output]:\n{action_text.strip()}")

        # Call AgentInstance.step
        step_states = {
            "observation_text": observation_text,
            "action_text": action_text,
            "label": str(task_idx),
            "sampling_params": None,
        }
        step_result = await instance.step(step_states)

        reward = step_result["rewards"].item()
        done = step_result["done"]
        feedback = step_result["environment_feedback"]
        total_reward += reward

        print(f"\n[Reward]: {reward}")
        print(f"[Done]: {done}")
        print(f"[Feedback (first 300 chars)]:\n{feedback[:300]}")

        # Update observation (same as MultiTurnAgentExecutor line 107)
        observation_text = observation_text + action_text + feedback

        if done:
            print(f"\n*** Episode finished at step {step + 1} ***")
            break

    print(f"\n{'='*70}")
    print(f"EPISODE SUMMARY")
    print(f"{'='*70}")
    print(f"  Task idx:     {task_idx}")
    print(f"  Steps taken:  {step + 1}")
    print(f"  Total reward: {total_reward}")
    print(f"  Final done:   {done}")
    print(f"  Context len:  {len(observation_text)} chars")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--webshop-url", default="http://localhost:6001")
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--task-idx", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()

    os.environ["WEBSHOP_SERVER_URL"] = args.webshop_url

    # Verify servers
    try:
        requests.get(f"{args.webshop_url}/health").raise_for_status()
        print("[OK] WebShop server is running")
    except Exception as e:
        print(f"[FAIL] WebShop server at {args.webshop_url}: {e}")
        sys.exit(1)

    try:
        requests.get(f"{args.vllm_url}/health").raise_for_status()
        print("[OK] vLLM server is running")
    except Exception as e:
        print(f"[FAIL] vLLM server at {args.vllm_url}: {e}")
        sys.exit(1)

    asyncio.run(run_episode(args.vllm_url, args.task_idx, args.max_steps))


if __name__ == "__main__":
    main()
