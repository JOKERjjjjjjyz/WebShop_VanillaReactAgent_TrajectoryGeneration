#!/usr/bin/env python3
"""
A/B Test: Thinking Mode On vs Off for Qwen3.5-4B on WebShop
"""
import json, os, re, sys, time, uuid
import urllib.request

PARG_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEBSHOP_PATH = os.environ.get("WEBSHOP_PATH", "/data1/yanze/WebShop")
for p in [PARG_ROOT, WEBSHOP_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-11-openjdk-amd64")

# Force unbuffered stdout
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

SYSTEM_PROMPT = """\
You are a shopping agent operating in a text-only WebShop environment.

## Action Space
- search[<keywords>] — submit a keyword search
- click[<exact text>] — click a button/link using EXACT text from page

## Output Format ← EVERY TURN
Thought: <reasoning>
Action: <one action>

## Rules
- Output ONLY Thought + Action. No extra text.
- "buy now" must be lower-case.
"""

FORMAT_RETRY = "Respond with ONLY:\nThought: <reason>\nAction: <action>"

def parse_response(text):
    thought, action = "", ""
    t = re.search(r'Thought:(.*?)(?=Action:|$)', text, re.DOTALL | re.IGNORECASE)
    a = re.search(r'Action:\s*(.*)', text, re.IGNORECASE)
    if t: thought = t.group(1).strip()
    if a: action = a.group(1).strip().split("\n")[0].strip().strip("'\"")
    for tok in ("buy now", "back to search", "next >", "< prev"):
        if action.lower() == tok:
            action = tok; break
    return thought, action

def llm_call(messages, enable_thinking, max_tokens=512):
    payload = json.dumps({
        "model": "Qwen3.5-4B",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:8000/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"] or ""
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        usage = data.get("usage", {})
        return content, usage.get("completion_tokens", 0)
    except Exception as e:
        print(f"    [LLM ERROR] {e}", flush=True)
        return f"[LLM error: {e}]", 0

def run_episode(env, task_idx, enable_thinking, max_steps=20):
    mode_str = "think=ON" if enable_thinking else "think=OFF"
    print(f"  Starting task {task_idx} ({mode_str})...", flush=True)

    obs_raw = env.reset(session=task_idx)
    if isinstance(obs_raw, tuple): obs_raw = obs_raw[0]

    if "[SEP]" in obs_raw:
        parts = obs_raw.split("[SEP]")
        instruction = parts[2].strip() if len(parts) > 2 else obs_raw
    else:
        instruction = obs_raw

    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Instruction: {instruction}\n\nObservation:\n{obs_raw}"},
    ]

    final_reward = 0.0
    total_gen_tokens = 0
    t0 = time.time()

    for step in range(max_steps):
        step_t0 = time.time()
        call_history = list(history)
        action = ""
        thought = ""

        for attempt in range(3):
            content, gen_tokens = llm_call(call_history, enable_thinking)
            total_gen_tokens += gen_tokens
            thought, action = parse_response(content)
            if action: break
            call_history = call_history + [
                {"role": "assistant", "content": content},
                {"role": "user",      "content": FORMAT_RETRY},
            ]

        step_dt = time.time() - step_t0

        if not action:
            print(f"    step {step+1}: NO ACTION (format fail) [{step_dt:.1f}s]", flush=True)
            break

        print(f"    step {step+1}: {action[:50]:<50s} [{step_dt:.1f}s]", flush=True)

        result = env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        if isinstance(obs, tuple): obs = obs[0]

        final_reward = float(reward)
        history.append({"role": "assistant", "content": f"Thought: {thought}\nAction: {action}"})
        history.append({"role": "user", "content": f"Observation:\n{obs}"})

        if done:
            break

    elapsed = time.time() - t0
    steps_taken = step + 1
    status = "✓ SUCCESS" if final_reward >= 1.0 else f"reward={final_reward:.2f}"
    print(f"  → task {task_idx} DONE: {status}  steps={steps_taken}  "
          f"gen_tok={total_gen_tokens}  time={elapsed:.1f}s\n", flush=True)

    return {
        "task_idx": task_idx, "reward": final_reward,
        "steps": steps_taken, "gen_tokens": total_gen_tokens,
        "elapsed_s": round(elapsed, 1), "is_success": final_reward >= 1.0,
    }

def main():
    NUM_TASKS = 10
    MAX_STEPS = 20
    TASKS = list(range(NUM_TASKS))

    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    print("Loading WebShop environment...", flush=True)
    env = WebAgentTextEnv(observation_mode="text", num_products=1000)
    print("Environment ready.\n", flush=True)

    print(f"{'═'*70}")
    print(f"  A/B Test: Thinking Mode (Qwen3.5-4B, {NUM_TASKS} tasks, max {MAX_STEPS} steps)")
    print(f"{'═'*70}\n", flush=True)

    # ── A: Thinking OFF ────────────────────────────────────────────
    print("▶ GROUP A: enable_thinking = False\n", flush=True)
    results_a = []
    for idx in TASKS:
        r = run_episode(env, idx, enable_thinking=False, max_steps=MAX_STEPS)
        results_a.append(r)

    # ── B: Thinking ON ─────────────────────────────────────────────
    print("▶ GROUP B: enable_thinking = True\n", flush=True)
    results_b = []
    for idx in TASKS:
        r = run_episode(env, idx, enable_thinking=True, max_steps=MAX_STEPS)
        results_b.append(r)

    # ── Comparison ─────────────────────────────────────────────────
    def s(results):
        n = len(results)
        succ = sum(r["is_success"] for r in results)
        return {
            "n": n, "success": succ, "pct": 100*succ/n,
            "avg_r": sum(r["reward"]/n for r in results),
            "avg_s": sum(r["steps"]/n for r in results),
            "avg_tok": sum(r["gen_tokens"]/n for r in results),
            "avg_t": sum(r["elapsed_s"]/n for r in results),
            "total_t": sum(r["elapsed_s"] for r in results),
        }

    sa, sb = s(results_a), s(results_b)

    print(f"\n{'═'*70}")
    print(f"  COMPARISON")
    print(f"{'═'*70}")
    print(f"{'Metric':<30s} {'Think OFF':>15s} {'Think ON':>15s}")
    print(f"{'─'*60}")
    print(f"{'Success rate':<30s} {sa['success']}/{sa['n']} ({sa['pct']:.0f}%){'':<5s} {sb['success']}/{sb['n']} ({sb['pct']:.0f}%)")
    print(f"{'Average reward':<30s} {sa['avg_r']:>15.4f} {sb['avg_r']:>15.4f}")
    print(f"{'Average steps':<30s} {sa['avg_s']:>15.2f} {sb['avg_s']:>15.2f}")
    print(f"{'Avg gen tokens/ep':<30s} {sa['avg_tok']:>15.0f} {sb['avg_tok']:>15.0f}")
    print(f"{'Avg time/episode (s)':<30s} {sa['avg_t']:>15.1f} {sb['avg_t']:>15.1f}")
    print(f"{'Total wall time (s)':<30s} {sa['total_t']:>15.1f} {sb['total_t']:>15.1f}")
    print(f"{'═'*70}", flush=True)

    if sb['avg_r'] > sa['avg_r'] + 0.05:
        print("\n  🏆 Thinking ON wins on reward.")
    elif sa['avg_r'] > sb['avg_r'] + 0.05:
        print("\n  🏆 Thinking OFF wins on reward.")
    else:
        print("\n  🤝 Similar rewards. Consider speed tradeoff.")

    out = os.path.join(PARG_ROOT, "reports", "ab_thinking_test.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"off": results_a, "on": results_b, "s_off": sa, "s_on": sb}, f, indent=2)
    print(f"  Saved: {out}\n", flush=True)

if __name__ == "__main__":
    main()
