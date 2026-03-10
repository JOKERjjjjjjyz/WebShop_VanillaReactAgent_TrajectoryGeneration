#!/usr/bin/env python3
"""
Full End-to-End Demo
=====================
Starts Qwen3.5-4B LLM server + WebShop env server, then runs
one complete ReAct episode with the vanilla agent.

Usage:
    cd /data1/yanze/PARG_WebStore
    CUDA_VISIBLE_DEVICES=0 \\
    /data1/yanze/miniconda3/envs/agent/bin/python scripts/run_demo.py

Flags:
    --task-idx INT     which WebShop task to run (default: random)
    --seed INT         random seed (default: 42)
    --max-steps INT    max steps per episode (default: 15)
    --llm-port INT     LLM server port (default: 8000)
    --env-port INT     WebShop env server port (default: 6001)
"""

import argparse
import os
import sys
import subprocess
import time
import requests
import json
import signal

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

AGENT_PYTHON  = "/data1/yanze/miniconda3/envs/agent/bin/python"
WEBSHOP_PYTHON = "/data1/yanze/miniconda3/envs/webshop/bin/python"
WEBSHOP_PATH   = "/data1/yanze/WebShop"
MODEL_PATH     = "/data1/yanze/.cache/huggingface/hub/qwen3.5-4b"


# ── Helpers ───────────────────────────────────────────────────────────────────

def wait_for_server(url: str, timeout: int = 120, label: str = "server") -> bool:
    print(f"  Waiting for {label} at {url} ...", end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print(" ✓")
                return True
        except Exception:
            pass
        time.sleep(2)
        print(".", end="", flush=True)
    print(" ✗ TIMEOUT")
    return False


def start_llm_server(port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["HF_HOME"] = "/data1/yanze/.cache/huggingface"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    proc = subprocess.Popen(
        [
            AGENT_PYTHON,
            os.path.join(ROOT, "runner", "local_llm_server.py"),
            "--model", MODEL_PATH,
            "--model-name", "Qwen3.5-4B",
            "--port", str(port),
            "--host", "0.0.0.0",
            "--dtype", "bfloat16",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def start_webshop_server(port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
    env["WEBSHOP_PATH"] = WEBSHOP_PATH

    proc = subprocess.Popen(
        [
            WEBSHOP_PYTHON,
            os.path.join(ROOT, "runner", "webshop_env_wrapper.py"),
            "--serve",
            "--port", str(port),
            "--num_products", "1000",
        ],
        env=env,
        cwd=WEBSHOP_PATH,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


# ── Main ──────────────────────────────────────────────────────────────────────

def run_demo(args):
    llm_proc = None
    env_proc = None

    def cleanup(sig=None, frame=None):
        print("\n[Demo] Shutting down servers...")
        if llm_proc: llm_proc.terminate()
        if env_proc: env_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # ── 1. Start servers ──────────────────────────────────────────────────────
    print("=" * 60)
    print("  PARG-WebStore End-to-End Demo")
    print("=" * 60)

    print("\n[Step 1/3] Starting LLM server (Qwen3.5-4B)...")
    llm_proc = start_llm_server(args.llm_port)
    if not wait_for_server(f"http://localhost:{args.llm_port}/health", timeout=150, label="LLM server"):
        print("ERROR: LLM server failed to start.")
        cleanup()

    print(f"\n[Step 2/3] Starting WebShop env server...")
    env_proc = start_webshop_server(args.env_port)
    if not wait_for_server(f"http://localhost:{args.env_port}/reset", timeout=90, label="WebShop server"):
        # POST /reset is the right endpoint — GET will 405 but that means server is up
        pass  # try anyway - it might 405 on GET but still be running
    time.sleep(3)  # give webshop a moment to finish loading products

    # ── 2. Init agent + env clients ───────────────────────────────────────────
    print(f"\n[Step 3/3] Running episode...\n")

    from runner.llm_client import OpenAIChatClient
    from runner.webshop_client import WebShopClient
    from runner.trajectory_logger import TrajectoryLogger

    llm  = OpenAIChatClient(
        model_name="Qwen3.5-4B",
        base_url=f"http://localhost:{args.llm_port}/v1",
        api_key="dummy",
        temperature=0.7,
        max_tokens=256,
    )
    env    = WebShopClient(base_url=f"http://localhost:{args.env_port}")
    logger = TrajectoryLogger(log_dir=os.path.join(ROOT, "logs", "raw", "webshop", "react_v1"))

    # ── 3. Run episode ────────────────────────────────────────────────────────
    import re
    import uuid

    SYSTEM_PROMPT = """\
You are an autonomous agent completing a shopping task on WebShop.
Given an instruction, search for products, explore results, select options, and buy the correct item.

Available Actions:
  search[<keywords>]    — search for products
  click[<text>]         — click a link or button using its EXACT text from the page

Format (STRICT — output ONLY this, nothing else):
Thought: <your brief reasoning. Reference earlier observations with [[obs:N]] if relevant>
Action: <one action, e.g. search[laptop bag] or click[B09QQWRW2M] or click[buy now]>

Episode ends when you click [buy now] or reach the step limit."""

    obs = env.reset(seed=args.seed, task_idx=args.task_idx)
    task_id = env.task_id
    episode_id = str(uuid.uuid4())

    # Extract the instruction from the initial obs
    instruction = obs
    if "[SEP]" in obs:
        # Obs format: "WebShop [SEP] Instruction: [SEP] <instruction text> [SEP] Search"
        parts = obs.split("[SEP]")
        instruction = parts[2].strip() if len(parts) > 2 else obs

    print("─" * 60)
    print(f"  Task ID   : {task_id}")
    print(f"  Episode ID: {episode_id}")
    print(f"  Instruction:\n  {instruction.strip()}")
    print("─" * 60)

    logger.start_episode(
        episode_id=episode_id,
        task_id=task_id,
        seed=args.seed or 42,
        policy_version="react_v1",
        toolset_version="webshop_text_v1",
    )

    # Log initial obs
    logger.log_node(
        node_type="OBS_TOOL",
        payload={"raw_text": obs, "tool_name": "env_reset"},
        call_id="step_0"
    )

    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Instruction: {instruction}\n\nObservation:\n{obs}"},
    ]

    success = False
    termination = "max_steps"
    final_reward = 0.0

    for step in range(args.max_steps):
        print(f"\n{'━'*60}")
        print(f"  Step {step + 1}/{args.max_steps}")
        print(f"{'━'*60}")

        # — LLM call —
        try:
            llm_output = llm.chat(history)
        except Exception as e:
            print(f"  [LLM ERROR] {e}")
            break

        # — Parse Thought / Action —
        thought, action = "", ""
        t_match = re.search(r'Thought:(.*?)(?:Action:|$)', llm_output, re.DOTALL | re.IGNORECASE)
        a_match = re.search(r'Action:(.*)', llm_output, re.IGNORECASE)
        if t_match:
            thought = t_match.group(1).strip()
        if a_match:
            action  = a_match.group(1).strip().split("\n")[0].strip()

        # Extract [[obs:N]] references
        ref_ids = [int(m.group(1)) for m in re.finditer(r'\[\[obs:(\d+)\]\]', thought)]

        print(f"  Thought : {thought[:200]}")
        print(f"  Action  : {action}")

        # — Log THOUGHT —
        import uuid as _uuid
        call_id = str(_uuid.uuid4())
        logger.log_node("THOUGHT", {"raw_text": thought}, ref_ids=ref_ids)
        logger.log_node("ACT_TOOL", {"action_str": action}, call_id=call_id)

        if not action:
            print("  [WARN] No action parsed, skipping step.")
            history.append({"role": "assistant", "content": llm_output})
            history.append({"role": "user", "content": "Please provide a valid Action."})
            continue

        # — Env step —
        try:
            obs, reward, done, info = env.step(action)
        except Exception as e:
            print(f"  [ENV ERROR] {e}")
            break

        final_reward = reward
        print(f"  Reward  : {reward:.4f}  Done: {done}")
        print(f"  Obs     : {obs[:250]}")

        # — Log OBS —
        logger.log_node(
            "OBS_TOOL",
            {"raw_text": obs, "tool_name": action.split("[")[0]},
            call_id=call_id
        )

        # — Update history —
        history.append({"role": "assistant", "content": llm_output})
        history.append({"role": "user", "content": f"Observation:\n{obs}"})

        if done:
            success = reward >= 1.0
            termination = "success" if success else "done_no_reward"
            break

    # ── 4. Episode summary ───────────────────────────────────────────────────
    logger.end_episode(
        is_success=success,
        total_cost=0.0,
        termination_reason=termination,
    )

    print(f"\n{'═'*60}")
    print(f"  EPISODE COMPLETE")
    print(f"  Success         : {success}")
    print(f"  Final reward    : {final_reward:.4f}")
    print(f"  Termination     : {termination}")
    print(f"  Steps taken     : {step + 1}")
    print(f"  Log written to  : logs/raw/webshop/react_v1/")
    print(f"{'═'*60}\n")

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a full PARG-WebStore demo episode")
    parser.add_argument("--task-idx",  type=int,   default=None,  help="WebShop task index (None = random)")
    parser.add_argument("--seed",      type=int,   default=42,    help="Random seed")
    parser.add_argument("--max-steps", type=int,   default=15,    help="Max steps per episode")
    parser.add_argument("--llm-port",  type=int,   default=8000,  help="LLM server port")
    parser.add_argument("--env-port",  type=int,   default=6001,  help="WebShop env server port")
    args = parser.parse_args()

    run_demo(args)
