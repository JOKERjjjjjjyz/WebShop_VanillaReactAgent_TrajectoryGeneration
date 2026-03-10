#!/usr/bin/env python3
"""
Parallel Baseline Rollout for PARG-WebStore (V2)
==================================================
Runs N tasks in parallel using threads.
Each worker owns a direct WebAgentTextEnv instance.
All workers share one vLLM server (Qwen3.5-4B on port 8000).

Features:
  - max_steps=1000 (configurable)
  - Per-step cumulative token count tracking
  - Failure-point detection and annotation
  - vLLM continuous batching compatible (8+ concurrent workers)

Prerequisites:
  Terminal 1 (agent env): ./scripts/start_llm_server.sh
  Terminal 2 (webshop env):
    JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \\
    WEBSHOP_PATH=/data1/yanze/WebShop \\
    /data1/yanze/miniconda3/envs/webshop/bin/python \\
        scripts/run_baseline.py --workers 8 --max-tasks 500 --max-steps 1000
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
PARG_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEBSHOP_PATH = os.environ.get("WEBSHOP_PATH", "/data1/yanze/WebShop")

for p in [PARG_ROOT, WEBSHOP_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-11-openjdk-amd64")

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a shopping agent operating in a text-only WebShop environment.
Your job: follow the instruction, browse the shop, and buy exactly the right product.

## Action Space
- search[<keywords>]       — submit a keyword search
- click[<exact text>]      — click an item or button using its EXACT text from the page
                             (product IDs like B09QQWRW2M, options like "large", nav like "buy now")

## Output Format  ← FOLLOW THIS EXACTLY, EVERY TURN
Thought: <one-sentence reasoning>
Action: <exactly one action, no extra text>

## Examples
Observation:
WebShop [SEP] Instruction: [SEP] Find me quick drying, moisture wicking women's activewear with long sleeve with color: b-purple-thumbhole, and size: x-large, and price lower than 30.00 dollars [SEP] Search
Thought: I need to find women's activewear matching the specific criteria: quick drying, moisture wicking, long sleeve, color b-purple-thumbhole, size x-large, and price under $30. I'll start with a search for long sleeve women's activewear.
Action: search[long sleeve women's activewear]

Observation:
Instruction: [SEP] Find me quick drying, moisture wicking women's activewear with long sleeve with color: b-purple-thumbhole, and size: x-large, and price lower than 30.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B08JD14GJF [SEP] COOrun Women's UPF 50+ Sun Protection Shirts Quick Dry Long Sleeve Shirts Lightweight T-Shirt Outdoor Hiking Runing Fishing [SEP] $9.98 to $16.98 [SEP] B08G14B779 [SEP] NIUTA 2 Pack Hair Towel Wrap, Microfiber Quick Drying Hair Towels, Super Absorbent Quick Dry Hair Towel, Wrapped Bath Cap (Pink+Light Blue) [SEP] $4.99 [SEP] B098824K9T [SEP] Spa Savvy Shower Cap and Twist Hair Turban Duo, Enhance Your Shower Experience with this Complete Set, Fully Lined Shower Cap, Microfiber Turban (Green) [SEP] $11.99 [SEP] B07WMMYB6G [SEP] MAGCOMSEN Men's Gym Workout Shorts with Pockets Mesh Liner Quick Dry Running Shorts for Jogging, Hiking [SEP] $17.98 to $17.99 [SEP] B09QCP4579 [SEP] LWLW High Waist Yoga Shorts for Women,Tummy Control Biker Shorts Exercise Workout Butt Lifting Tights Women's Short Pants [SEP] $13.99 [SEP] B096P77XMD [SEP] Runhit 2 in 1 Men’s Running Shorts 5" Quick Dry Workout Shorts for Men with Pockets Gym Athletic Shorts [SEP] $14.98 to $17.98 [SEP] B09RK77R3V [SEP] Gym Shorts For Men Quick Dry Lightweight Training Running Jogger Drawstring Pockets Solid Zipper Shorts [SEP] $8.29 to $10.79 [SEP] B0065YIB8I [SEP] Body Drench Quick Tan Instant Self-Tanner, Bronzing Mousse, Medium Dark, 4.2 oz [SEP] $10.0 [SEP] B09NLZFMRG [SEP] ONGHAHYIS Shower Scrubbing Mitt Gloves Body Brush Take A Shower in 1 Minute Exfoliating Scrub Foot Scrubber Cath Sponge High-Elastic Five-Finger Bath Towel (Color : Random Color 1pc) [SEP] $22.16 [SEP] B09PFWVXVS [SEP] Micvtve Key Holder for Wall Mount Sweet Home Organizer Decorative, Metal Wall Coat Rack Key Hanger for Front Door,Kitchen Holder [SEP] $16.25
Thought: B08JD14GJF mentions "Quick Dry Long Sleeve Shirts" and is for Women, making it a strong candidate to check for the specific color and size options.
Action: click[B08JD14GJF]

Observation:
Instruction: [SEP] Find me quick drying, moisture wicking women's activewear with long sleeve with color: b-purple-thumbhole, and size: x-large, and price lower than 30.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] b-orange-thumbhole [SEP] b-sky blue-thumbhole [SEP] c-black [SEP] c-grey [SEP] a-white-thumbhole [SEP] b-green-thumbhole [SEP] c-blue [SEP] c-navy blue [SEP] c-purple [SEP] c-white [SEP] b-black-thumbhole [SEP] b-light grey-thumbhole [SEP] b-peach-thumbhole [SEP] c-green [SEP] b-navy blue-thumbhole [SEP] b-purple-thumbhole [SEP] b-grey-thumbhole [SEP] size [SEP] small [SEP] medium [SEP] large [SEP] x-large [SEP] xx-large [SEP] COOrun Women's UPF 50+ Sun Protection Shirts Quick Dry Long Sleeve Shirts Lightweight T-Shirt Outdoor Hiking Runing Fishing [SEP] Price: $9.98 to $16.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now
Thought: I am on the product detail page for B08JD14GJF. This item is women's, long sleeve, quick dry, and priced under $30. I need to select the color "b-purple-thumbhole".
Action: click[b-purple-thumbhole]

Observation:
Instruction: [SEP] Find me quick drying, moisture wicking women's activewear with long sleeve with color: b-purple-thumbhole, and size: x-large, and price lower than 30.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] b-orange-thumbhole [SEP] b-sky blue-thumbhole [SEP] c-black [SEP] c-grey [SEP] a-white-thumbhole [SEP] b-green-thumbhole [SEP] c-blue [SEP] c-navy blue [SEP] c-purple [SEP] c-white [SEP] b-black-thumbhole [SEP] b-light grey-thumbhole [SEP] b-peach-thumbhole [SEP] c-green [SEP] b-navy blue-thumbhole [SEP] b-purple-thumbhole [SEP] b-grey-thumbhole [SEP] size [SEP] small [SEP] medium [SEP] large [SEP] x-large [SEP] xx-large [SEP] COOrun Women's UPF 50+ Sun Protection Shirts Quick Dry Long Sleeve Shirts Lightweight T-Shirt Outdoor Hiking Runing Fishing [SEP] Price: $9.98 to $16.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now
Thought: I have selected the correct color. Now I need to select the size "x-large".
Action: click[x-large]

Observation:
Instruction: [SEP] Find me quick drying, moisture wicking women's activewear with long sleeve with color: b-purple-thumbhole, and size: x-large, and price lower than 30.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] b-orange-thumbhole [SEP] b-sky blue-thumbhole [SEP] c-black [SEP] c-grey [SEP] a-white-thumbhole [SEP] b-green-thumbhole [SEP] c-blue [SEP] c-navy blue [SEP] c-purple [SEP] c-white [SEP] b-black-thumbhole [SEP] b-light grey-thumbhole [SEP] b-peach-thumbhole [SEP] c-green [SEP] b-navy blue-thumbhole [SEP] b-purple-thumbhole [SEP] b-grey-thumbhole [SEP] size [SEP] small [SEP] medium [SEP] large [SEP] x-large [SEP] xx-large [SEP] COOrun Women's UPF 50+ Sun Protection Shirts Quick Dry Long Sleeve Shirts Lightweight T-Shirt Outdoor Hiking Runing Fishing [SEP] Price: $9.98 to $16.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now
Thought: I have selected the correct color and size. I will click "Buy Now" to complete the purchase.
Action: click[buy now]


Observation:
WebShop [SEP] Instruction: [SEP] Find me mens dress shirt cotton spandex classic fit short sleeve black xx-large tall, and price lower than 50.00 dollars [SEP] Search
Thought: I need to find a men's dress shirt that is cotton spandex, classic fit, short sleeve, black, size xx-large tall, and under $50. I will start by searching for these exact keywords.
Action: search[mens dress shirt cotton spandex classic fit short sleeve black xx-large tall]

Observation:
Instruction: [SEP] Find me mens dress shirt cotton spandex classic fit short sleeve black xx-large tall, and price lower than 50.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 0) [SEP]

[System Warning: identical_obs. Your previous action was ineffective or repeated. Please try a DIFFERENT strategy or look carefully at the available options.]
Thought: My search returned 0 results. The search terms were too specific. I should broaden my search to just "men's dress shirts" and then filter by the features later.
Action: search[men's dress shirts]

## Rules
- Output ONLY the Thought + Action block. No extra commentary.
- Do not repeat an action that already returned the same page.
- "buy now" and navigation text must be lower-case as shown in the page.
"""

FORMAT_RETRY_MSG = ("Your last response did not contain an Action line.\n"
                    "Respond with ONLY:\nThought: <reason>\nAction: <action>")


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_response(text: str):
    thought, action = "", ""
    t = re.search(r'Thought:(.*?)(?=Action:|$)', text, re.DOTALL | re.IGNORECASE)
    a = re.search(r'Action:\s*(.*)', text, re.IGNORECASE)
    if t:
        thought = t.group(1).strip()
    if a:
        action = a.group(1).strip().split("\n")[0].strip().strip("'\"")
    # Normalise navigation tokens
    for tok in ("buy now", "back to search", "next >", "< prev"):
        if action.lower() == tok:
            action = tok
            break
    refs = [int(m.group(1)) for m in re.finditer(r'\[\[obs:(\d+)\]\]', thought)]
    return thought, action, refs


# ── Failure-point detection ───────────────────────────────────────────────────

def detect_failure(
    action: str,
    obs: str,
    prev_obs: str,
    prev_action: str,
    was_on_product_page: bool,
) -> tuple:
    """
    Heuristically detect whether this step represents a failure point.

    Returns (failure_flag: bool, failure_reason: str or None).
    A failure point = a step where the agent made no useful progress.
    """
    # 1. Identical observation (action had no effect)
    if obs == prev_obs and obs:
        return True, "identical_obs"

    # 2. Agent went backwards after being on a product page
    if was_on_product_page and action.lower() == "back to search":
        return True, "backtrack_from_product"

    # 3. Action was invalid → WebShop usually returns the search page
    # Check if agent clicked something and ended up on the search page unexpectedly
    if action.startswith("click[") and "[SEP] Search" in obs and "[SEP] Search" not in prev_obs:
        return True, "invalid_click_to_search"

    # 4. Search returned the same results page (no new information)
    if action.startswith("search[") and prev_action.startswith("search["):
        # Two consecutive searches — the first might have been unhelpful
        return True, "consecutive_search"

    return False, None


def is_product_page(obs: str) -> bool:
    """Check if the observation looks like a product detail page."""
    return "Buy Now" in obs or "buy now" in obs or "[SEP] Description" in obs


# ── Token counting ────────────────────────────────────────────────────────────

def count_context_tokens(history: list) -> int:
    """
    Estimate context length by counting characters / 3.5.
    This avoids loading the tokenizer in the webshop env (Python 3.8).
    A more precise count uses the vLLM /tokenize endpoint, but
    char-based estimation is fast and sufficient for tracking trends.
    """
    total_chars = sum(len(m.get("content", "")) for m in history)
    return int(total_chars / 3.5)  # rough char-to-token ratio for English


# ── Thread-safe JSONL logger ──────────────────────────────────────────────────

class EpisodeLogger:
    """Writes one JSONL record per episode, thread-safe."""
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict):
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── LLM HTTP client ──────────────────────────────────────────────────────────

class LLMClient:
    """Lightweight HTTP client for vLLM / OpenAI-compatible server."""
    def __init__(self, base_url="http://localhost:8000/v1",
                 model="Qwen3.5-4B", temperature=0.7,
                 max_tokens=512, timeout=120):
        self.base_url    = base_url.rstrip("/")
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout

    def chat(self, messages: list) -> str:
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                content = data["choices"][0]["message"]["content"]
                return content if content else ""
            except Exception as e:
                if attempt == 2:
                    return f"[LLM error: {e}]"
                time.sleep(2)
        return "[LLM error: max retries]"


# ── Per-worker episode runner ─────────────────────────────────────────────────

def run_one_episode(env, task_idx, llm, episode_logger,
                    max_steps=1000, max_format_retries=2):
    """Run a single episode and return result stats."""

    obs_raw = env.reset(session=task_idx)
    if isinstance(obs_raw, tuple):
        obs_raw = obs_raw[0]

    if "[SEP]" in obs_raw:
        parts = obs_raw.split("[SEP]")
        instruction = parts[2].strip() if len(parts) > 2 else obs_raw
    else:
        instruction = obs_raw

    task_id    = f"webshop_{task_idx}"
    episode_id = str(uuid.uuid4())
    trajectory = []
    failure_points = []

    def log_step(**kwargs):
        kwargs["t"] = time.time()
        trajectory.append(kwargs)

    log_step(node_type="OBS_TOOL",
             payload={"raw_text": obs_raw, "tool_name": "env_reset"},
             call_id="init", context_tokens=0)

    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Instruction: {instruction}\n\nObservation:\n{obs_raw}"},
    ]

    final_reward  = 0.0
    termination   = "max_steps"
    steps_taken   = 0
    format_errors = 0
    prev_obs      = ""
    prev_action   = ""

    for step in range(max_steps):
        steps_taken = step + 1

        # ── LLM call with format-retry ────────────────────────────────
        call_history = list(history)
        action = ""
        thought = ""
        refs = []

        for attempt in range(max_format_retries + 1):
            raw = llm.chat(call_history)
            thought, action, refs = parse_response(raw)
            if action:
                break
            format_errors += 1
            if attempt < max_format_retries:
                call_history = call_history + [
                    {"role": "assistant", "content": raw},
                    {"role": "user",      "content": FORMAT_RETRY_MSG},
                ]

        ctx_tokens = count_context_tokens(history)
        call_id = str(uuid.uuid4())

        log_step(node_type="THOUGHT", payload={"raw_text": thought},
                 ref_ids=refs, context_tokens=ctx_tokens)
        log_step(node_type="ACT_TOOL", payload={"action_str": action},
                 call_id=call_id, context_tokens=ctx_tokens)

        if not action:
            termination = "format_failure"
            break

        # ── Env step ──────────────────────────────────────────────────
        try:
            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
            if isinstance(obs, tuple):
                obs = obs[0]
        except Exception as e:
            obs, reward, done = f"[env_error: {e}]", 0.0, True
            termination = "env_error"

        final_reward = float(reward)

        # ── Failure-point detection ───────────────────────────────────
        was_on_product = is_product_page(prev_obs) if prev_obs else False
        fail_flag, fail_reason = detect_failure(
            action, obs, prev_obs, prev_action, was_on_product
        )
        if fail_flag:
            failure_points.append({
                "step": steps_taken,
                "action": action,
                "reason": fail_reason,
            })

        tool_name = action.split("[")[0] if "[" in action else action
        log_step(node_type="OBS_TOOL",
                 payload={"raw_text": obs, "tool_name": tool_name},
                 call_id=call_id, context_tokens=ctx_tokens,
                 failure_flag=fail_flag, failure_reason=fail_reason)

        history.append({"role": "assistant",
                        "content": f"Thought: {thought}\nAction: {action}"})
        
        obs_content = f"Observation:\n{obs}"
        if fail_flag:
            obs_content += f"\n\n[System Warning: {fail_reason}. Your previous action was ineffective or repeated. Please try a DIFFERENT strategy or look carefully at the available options.]"
            
        history.append({"role": "user", "content": obs_content})

        prev_obs    = obs
        prev_action = action

        if done:
            termination = "done"
            break

    is_success = final_reward >= 1.0
    is_partial = (not is_success) and (final_reward > 0.0)
    final_ctx_tokens = count_context_tokens(history)

    episode_logger.write({
        "episode_id":       episode_id,
        "task_id":          task_id,
        "task_idx":         task_idx,
        "instruction":      instruction,
        "steps":            steps_taken,
        "final_reward":     final_reward,
        "is_success":       is_success,
        "is_partial":       is_partial,
        "termination":      termination,
        "format_errors":    format_errors,
        "final_ctx_tokens": final_ctx_tokens,
        "failure_points":   failure_points,
        "num_failures":     len(failure_points),
        "trajectory":       trajectory,
    })

    return {
        "task_idx":         task_idx,
        "task_id":          task_id,
        "is_success":       is_success,
        "is_partial":       is_partial,
        "reward":           final_reward,
        "steps":            steps_taken,
        "termination":      termination,
        "format_errors":    format_errors,
        "final_ctx_tokens": final_ctx_tokens,
        "num_failures":     len(failure_points),
    }


# ── Worker thread ─────────────────────────────────────────────────────────────

def worker_fn(task_queue, llm, episode_logger, max_steps, results, lock, progress):
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    env = WebAgentTextEnv(observation_mode="text", num_products=1000)

    while True:
        with lock:
            if not task_queue:
                break
            task_idx = task_queue.pop(0)

        try:
            result = run_one_episode(
                env=env, task_idx=task_idx, llm=llm,
                episode_logger=episode_logger, max_steps=max_steps,
            )
        except Exception as e:
            result = {
                "task_idx": task_idx, "task_id": f"webshop_{task_idx}",
                "is_success": False, "is_partial": False,
                "reward": 0.0, "steps": 0,
                "termination": f"crash: {e}", "format_errors": 0,
                "final_ctx_tokens": 0, "num_failures": 0,
            }

        with lock:
            results.append(result)
            done_n = len(results)
            total  = progress["total"]
            succ   = sum(r["is_success"] for r in results)
            part   = sum(r["is_partial"] for r in results)
            avg_r  = sum(r["reward"] for r in results) / done_n
            avg_tk = sum(r["final_ctx_tokens"] for r in results) / done_n
            elapsed = time.time() - progress["t0"]
            eta    = (elapsed / done_n) * (total - done_n) if done_n > 0 else 0
            print(
                f"\r  [{done_n:4d}/{total}]  "
                f"success={succ} ({100*succ/done_n:.1f}%)  "
                f"partial={part}  "
                f"avg_r={avg_r:.3f}  "
                f"avg_tok={avg_tk:.0f}  "
                f"ETA={eta/60:.1f}min   ",
                end="", flush=True,
            )


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(results, report_dir, args):
    os.makedirs(report_dir, exist_ok=True)
    n = len(results)
    if n == 0:
        print("\nNo results to report.")
        return

    successes = [r for r in results if r["is_success"]]
    partials  = [r for r in results if r["is_partial"]]
    failures  = [r for r in results if not r["is_success"] and not r["is_partial"]]

    avg_reward   = sum(r["reward"] for r in results) / n
    avg_steps    = sum(r["steps"]  for r in results) / n
    avg_fmt_err  = sum(r["format_errors"] for r in results) / n
    avg_tokens   = sum(r["final_ctx_tokens"] for r in results) / n
    max_tokens   = max(r["final_ctx_tokens"] for r in results)
    min_tokens   = min(r["final_ctx_tokens"] for r in results)
    avg_failures = sum(r["num_failures"] for r in results) / n
    total_fails  = sum(r["num_failures"] for r in results)

    term_counts = {}
    for r in results:
        term_counts[r["termination"]] = term_counts.get(r["termination"], 0) + 1

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── JSONL summary ─────────────────────────────────────────────────
    summary_jsonl = os.path.join(report_dir, "baseline_react.jsonl")
    with open(summary_jsonl, "w", encoding="utf-8") as f:
        for r in sorted(results, key=lambda x: x["task_idx"]):
            f.write(json.dumps(r) + "\n")

    # ── Markdown report ───────────────────────────────────────────────
    md = [
        "# Baseline ReAct Report (V2)",
        "",
        f"**Generated**: {timestamp}  ",
        f"**Model**: Qwen3.5-4B (vLLM 0.17.0, `bfloat16`, single GPU)  ",
        f"**Environment**: WebShop text mode, `num_products=1000`  ",
        f"**Policy**: vanilla ReAct, `policy_version={args.policy_version}`  ",
        f"**Tasks sampled**: {n} / 6910 total goals  ",
        f"**Max steps / episode**: {args.max_steps}  ",
        f"**Parallel workers**: {args.workers}  ",
        "",
        "---",
        "",
        "## Results Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Full success** (reward = 1.0) | {len(successes)} / {n} = **{100*len(successes)/n:.1f}%** |",
        f"| **Partial success** (0 < reward < 1) | {len(partials)} / {n} = **{100*len(partials)/n:.1f}%** |",
        f"| **Failure** (reward = 0) | {len(failures)} / {n} = **{100*len(failures)/n:.1f}%** |",
        f"| **Average reward** | **{avg_reward:.4f}** |",
        f"| **Average steps** | {avg_steps:.2f} |",
        f"| **Average format errors/episode** | {avg_fmt_err:.2f} |",
        "",
        "## Token Usage",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Average final context tokens** | {avg_tokens:.0f} |",
        f"| **Min context tokens** | {min_tokens} |",
        f"| **Max context tokens** | {max_tokens} |",
        "",
        "## Failure-Point Analysis",
        "",
        f"Total failure points detected across all episodes: **{total_fails}**  ",
        f"Average failure points per episode: **{avg_failures:.2f}**  ",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]

    # Count failure reasons from JSONL log (we only have summary stats here)
    md.append(f"| Episodes with ≥1 failure point | {sum(1 for r in results if r['num_failures'] > 0)} ({100*sum(1 for r in results if r['num_failures'] > 0)/n:.1f}%) |")

    md += [
        "",
        "## Termination Breakdown",
        "",
        "| Reason | Count |",
        "|--------|-------|",
    ]
    for k, v in sorted(term_counts.items(), key=lambda x: -x[1]):
        md.append(f"| `{k}` | {v} ({100*v/n:.1f}%) |")

    md += [
        "",
        "## Reward Distribution",
        "",
        "| Reward bucket | Count | % |",
        "|---|---|---|",
    ]
    buckets = {"0.0": 0, "(0, 0.5)": 0, "[0.5, 1.0)": 0, "1.0": 0}
    for r in results:
        rw = r["reward"]
        if rw == 0.0:      buckets["0.0"] += 1
        elif rw < 0.5:     buckets["(0, 0.5)"] += 1
        elif rw < 1.0:     buckets["[0.5, 1.0)"] += 1
        else:              buckets["1.0"] += 1
    for bk, bv in buckets.items():
        md.append(f"| `{bk}` | {bv} | {100*bv/n:.1f}% |")

    md += [
        "",
        "## Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        f"| `logs/raw/webshop/react_v1/<date>_baseline.jsonl` | Full trajectories with token counts + failure annotations |",
        f"| `reports/baseline_react.jsonl` | Summary stats per episode (no trajectory) |",
        f"| `reports/baseline_react.md` | This report |",
    ]

    md_path = os.path.join(report_dir, "baseline_react.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n\n{'═'*62}")
    print(f"  BASELINE COMPLETE")
    print(f"{'═'*62}")
    print(f"  Tasks:           {n}")
    print(f"  Full success:    {len(successes)} ({100*len(successes)/n:.1f}%)")
    print(f"  Partial:         {len(partials)} ({100*len(partials)/n:.1f}%)")
    print(f"  Avg reward:      {avg_reward:.4f}")
    print(f"  Avg steps:       {avg_steps:.2f}")
    print(f"  Avg ctx tokens:  {avg_tokens:.0f} (min={min_tokens}, max={max_tokens})")
    print(f"  Failure points:  {total_fails} total, {avg_failures:.2f}/episode")
    print(f"  Report:          {md_path}")
    print(f"  Summary JSONL:   {summary_jsonl}")
    print(f"{'═'*62}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",        type=int,   default=8)
    parser.add_argument("--max-tasks",      type=int,   default=500)
    parser.add_argument("--max-steps",      type=int,   default=1000)
    parser.add_argument("--llm-url",        type=str,   default="http://localhost:8000/v1")
    parser.add_argument("--llm-model",      type=str,   default="Qwen3.5-4B")
    parser.add_argument("--temperature",    type=float, default=0.7)
    parser.add_argument("--task-start",     type=int,   default=0)
    parser.add_argument("--policy-version", type=str,   default="react_v1")
    args = parser.parse_args()

    date_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path   = os.path.join(PARG_ROOT, "logs", "raw", "webshop", "react_v1",
                              f"{date_str}_baseline.jsonl")
    report_dir = os.path.join(PARG_ROOT, "reports")

    llm = LLMClient(
        base_url=args.llm_url, model=args.llm_model,
        temperature=args.temperature, max_tokens=512, timeout=120,
    )
    episode_logger = EpisodeLogger(log_path)

    all_tasks = list(range(args.task_start, args.task_start + args.max_tasks))

    print(f"\n{'═'*62}")
    print(f"  PARG-WebStore Baseline Rollout V2")
    print(f"{'═'*62}")
    print(f"  Tasks:      {len(all_tasks)} (idx {all_tasks[0]}–{all_tasks[-1]})")
    print(f"  Workers:    {args.workers}")
    print(f"  Max steps:  {args.max_steps}")
    print(f"  LLM:        {args.llm_url}  model={args.llm_model}")
    print(f"  Log:        {log_path}")
    print(f"{'═'*62}\n")

    import multiprocessing

    with multiprocessing.Manager() as manager:
        task_queue = manager.list(all_tasks)
        results    = manager.list()
        lock       = manager.Lock()
        progress   = manager.dict({"total": len(all_tasks), "t0": time.time()})

        processes = []
        for _ in range(args.workers):
            p = multiprocessing.Process(
                target=worker_fn,
                args=(task_queue, llm, episode_logger, args.max_steps,
                      results, lock, progress),
                daemon=True,
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Convert managed list back to normal list for reporting
        results = list(results)

    print()
    generate_report(results, report_dir, args)


if __name__ == "__main__":
    main()
