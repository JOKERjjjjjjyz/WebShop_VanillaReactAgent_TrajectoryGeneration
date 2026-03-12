"""
Smoke test for WebShop pool server + AgentInstance.

Usage:
    1. Start the pool server first (conda: webshop):
       bash scripts/start_webshop_pool_server.sh 6001

    2. Run this test (conda: agent):
       python scripts/smoke_test_pool_server.py [--url http://localhost:6001]
"""

import argparse
import asyncio
import json
import sys
import time

import requests


def test_server_http(base_url):
    """Test the pool server HTTP endpoints synchronously."""
    print("=" * 60)
    print("Phase 1: HTTP endpoint tests")
    print("=" * 60)

    # Health check
    print("\n[1/5] Health check...")
    resp = requests.get(f"{base_url}/health")
    resp.raise_for_status()
    health = resp.json()
    print(f"  OK: {health}")

    # Create session with specific task
    print("\n[2/5] Create session (task_idx=0)...")
    resp = requests.post(f"{base_url}/create_session", json={"task_idx": 0})
    resp.raise_for_status()
    data = resp.json()
    session_id = data["session_id"]
    obs = data["obs"]
    instruction = data["instruction_text"]
    print(f"  session_id: {session_id[:16]}...")
    print(f"  instruction: {instruction[:100]}...")
    print(f"  obs (first 200 chars): {obs[:200]}...")

    # Step: search
    print("\n[3/5] Step: search[laptop]...")
    resp = requests.post(f"{base_url}/step", json={
        "session_id": session_id,
        "action": "search[laptop]",
    })
    resp.raise_for_status()
    step_data = resp.json()
    print(f"  reward: {step_data['reward']}, done: {step_data['done']}")
    print(f"  obs (first 200 chars): {step_data['obs'][:200]}...")

    # Create a second session concurrently
    print("\n[4/5] Create second session (task_idx=1) to test concurrency...")
    resp2 = requests.post(f"{base_url}/create_session", json={"task_idx": 1})
    resp2.raise_for_status()
    data2 = resp2.json()
    session_id2 = data2["session_id"]
    print(f"  session_id_2: {session_id2[:16]}...")
    print(f"  instruction_2: {data2['instruction_text'][:100]}...")

    # Check health shows 2 sessions (first one might have been closed if done)
    resp = requests.get(f"{base_url}/health")
    health = resp.json()
    print(f"  active_sessions: {health['active_sessions']}")

    # Close sessions
    print("\n[5/5] Close sessions...")
    requests.post(f"{base_url}/close_session", json={"session_id": session_id})
    requests.post(f"{base_url}/close_session", json={"session_id": session_id2})
    resp = requests.get(f"{base_url}/health")
    print(f"  active_sessions after cleanup: {resp.json()['active_sessions']}")

    print("\n  Phase 1 PASSED")


async def test_agent_instance(base_url):
    """Test the AgentInstance class directly."""
    print("\n" + "=" * 60)
    print("Phase 2: AgentInstance tests")
    print("=" * 60)

    import os
    os.environ["WEBSHOP_SERVER_URL"] = base_url

    # Import agent_func_webshop directly to avoid full openrlhf import chain
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "agent_func_webshop",
        "/data1/yanze/OpenRLHF/examples/python/agent_func_webshop.py",
        submodule_search_locations=[],
    )
    # Stub out openrlhf dependency to avoid importing the full package
    import types
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

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    AgentInstance = mod.AgentInstance
    parse_action = mod.parse_action

    # Test parse_action
    print("\n[1/3] Test parse_action...")
    assert parse_action("Thought: testing\nAction: search[laptop bag]") == "search[laptop bag]"
    assert parse_action("Thought: found it\nAction: click[Buy Now]") == "click[buy now]"
    assert parse_action("Thought: go back\nAction: click[Back to Search]") == "click[back to search]"
    assert parse_action("no action here") == ""
    print("  OK: all parse tests passed")

    # Test AgentInstance reset
    print("\n[2/3] Test AgentInstance.reset()...")
    # async __init__ - calling the class returns a coroutine in normal Python,
    # but OpenRLHF calls it synchronously. For testing, manually init.
    instance = AgentInstance.__new__(AgentInstance)
    instance.step_idx = 0
    instance.max_steps = 15
    instance.session_id = None
    instance._http_session = None

    states = {"observation": "placeholder", "label": "0"}
    result = await instance.reset(states)
    obs_text = result["observation"]
    print(f"  observation starts with: {obs_text[:80]}...")
    assert "<|im_start|>system" in obs_text, "Missing system tag"
    assert "<|im_start|>user" in obs_text, "Missing user tag"
    assert "Instruction:" in obs_text, "Missing instruction"
    assert "<|im_start|>assistant" in obs_text, "Missing assistant tag"
    print("  OK: reset returned properly formatted observation")

    # Test AgentInstance step
    print("\n[3/3] Test AgentInstance.step()...")
    action_text = "Thought: I need to search for the product.\nAction: search[laptop bag]"
    step_states = {
        "observation_text": obs_text,
        "action_text": action_text,
        "label": "0",
        "sampling_params": None,
    }
    step_result = await instance.step(step_states)
    print(f"  rewards: {step_result['rewards'].item()}")
    print(f"  done: {step_result['done']}")
    print(f"  feedback (first 200 chars): {step_result['environment_feedback'][:200]}...")
    assert "environment_feedback" in step_result
    assert "rewards" in step_result
    assert "done" in step_result
    print("  OK: step returned valid result")

    print("\n  Phase 2 PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:6001")
    args = parser.parse_args()

    print(f"Testing WebShop pool server at {args.url}\n")

    # Phase 1: HTTP tests
    try:
        test_server_http(args.url)
    except requests.ConnectionError:
        print(f"\n  FAILED: Cannot connect to {args.url}")
        print("  Make sure the pool server is running:")
        print("    bash scripts/start_webshop_pool_server.sh 6001")
        sys.exit(1)

    # Phase 2: AgentInstance tests
    asyncio.run(test_agent_instance(args.url))

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
