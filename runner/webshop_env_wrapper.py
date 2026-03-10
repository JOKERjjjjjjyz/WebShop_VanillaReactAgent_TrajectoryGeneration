"""
WebShop Environment Wrapper for PARG_WebStore
=============================================
Wraps the princeton-nlp/WebShop text environment (WebAgentTextEnv)
into a clean interface compatible with agentbench_wrapper.py (env.step / env.reset).

Usage:
    env = WebShopEnv(webshop_path="/data1/yanze/WebShop", num_products=1000)
    obs = env.reset(task_idx=0)   # or randomly sampled
    obs, reward, done, info = env.step("search[laptop bag]")
    obs, reward, done, info = env.step("click[B09QQWRW2M]")
    obs, reward, done, info = env.step("click[buy now]")

Notes:
    - Uses the 'text' observation mode (no browser/selenium needed).
    - Requires the webshop conda env python, OR install webshop deps in agent env.
    - Java must be reachable (JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64).
    - For cross-env calls, run this file as an HTTP server (see --serve flag).
"""

import sys
import os
import json
import random

# How to find WebShop — adjust if needed
WEBSHOP_PATH = os.environ.get("WEBSHOP_PATH", "/data1/yanze/WebShop")

if WEBSHOP_PATH not in sys.path:
    sys.path.insert(0, WEBSHOP_PATH)

os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-11-openjdk-amd64")


class WebShopEnv:
    """
    A thin wrapper around WebAgentTextEnv that provides:
      - env.reset(task_idx=None) -> str  (observation text)
      - env.step(action_str)     -> (str, float, bool, dict)
      - env.seed attribute for TrajectoryLogger compatibility
    """

    def __init__(self, num_products: int = 1000, webshop_path: str = WEBSHOP_PATH):
        if webshop_path not in sys.path:
            sys.path.insert(0, webshop_path)

        # Lazy import — only resolves after sys.path is set
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

        self._env = WebAgentTextEnv(observation_mode="text", num_products=num_products)
        self.seed = 42  # default; overridden in reset()
        self._task_idx = None
        self._num_products = num_products

    def reset(self, task_idx: int = None, seed: int = None) -> str:
        """
        Reset the environment.

        Args:
            task_idx: which task to load (int). None = random.
            seed:     random seed for task sampling. Sets self.seed.

        Returns:
            obs string
        """
        if seed is not None:
            self.seed = seed
            random.seed(seed)

        if task_idx is not None:
            self._task_idx = task_idx
            obs = self._env.reset(session=task_idx)
        else:
            obs = self._env.reset()
            self._task_idx = self._env.session

        # WebAgentTextEnv.reset() returns (text, info) or just text depending on version
        if isinstance(obs, tuple):
            obs = obs[0]

        return obs

    def step(self, action: str):
        """
        Take an action.

        Args:
            action: e.g. "search[query]", "click[element]"

        Returns:
            (obs: str, reward: float, done: bool, info: dict)
        """
        result = self._env.step(action)

        # Unpack flexibly — older gym returns 4-tuple, newer 5-tuple (truncated)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result

        if isinstance(obs, tuple):
            obs = obs[0]

        return obs, float(reward), bool(done), info or {}

    @property
    def task_id(self) -> str:
        return f"webshop_{self._task_idx}"

    def get_available_actions(self):
        """Return the available actions dict from the env."""
        return self._env.get_available_actions()


# ──────────────────────────────────────────────────────────────────────────────
# Optional: expose as an HTTP server so that the agent env (Python 3.10/CUDA)
# can call into the webshop env (Python 3.8) over a simple REST interface.
# Usage: python webshop_env_wrapper.py --serve --port 6001
# ──────────────────────────────────────────────────────────────────────────────

def run_server(port: int = 6001, num_products: int = 1000):
    """
    A super-minimal Flask server that wraps WebShopEnv.
    Endpoints:
        POST /reset   body: {"task_idx": int (optional), "seed": int (optional)}
        POST /step    body: {"action": str}
        GET  /actions
    """
    from flask import Flask, request, jsonify

    app = Flask(__name__)
    env = WebShopEnv(num_products=num_products)

    @app.route("/reset", methods=["POST"])
    def reset():
        data = request.get_json(force=True, silent=True) or {}
        obs = env.reset(
            task_idx=data.get("task_idx"),
            seed=data.get("seed"),
        )
        return jsonify({"obs": obs, "task_id": env.task_id, "seed": env.seed})

    @app.route("/step", methods=["POST"])
    def step():
        data = request.get_json(force=True, silent=True) or {}
        action = data.get("action", "")
        obs, reward, done, info = env.step(action)
        return jsonify({"obs": obs, "reward": reward, "done": done, "info": info})

    @app.route("/actions", methods=["GET"])
    def actions():
        return jsonify(env.get_available_actions())

    print(f"[WebShopServer] Serving on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Run as HTTP server")
    parser.add_argument("--port", type=int, default=6001)
    parser.add_argument("--num_products", type=int, default=1000)
    parser.add_argument("--test", action="store_true", help="Run a quick smoke test")
    args = parser.parse_args()

    if args.serve:
        run_server(port=args.port, num_products=args.num_products)
    elif args.test:
        print("=== WebShop Smoke Test ===")
        env = WebShopEnv(num_products=args.num_products)
        obs = env.reset(seed=42)
        print(f"Task ID: {env.task_id}")
        print(f"Obs (first 300 chars):\n{obs[:300]}\n")

        obs2, r2, done2, info2 = env.step("search[laptop bag]")
        print(f"After search: reward={r2}, done={done2}")
        print(f"Obs (first 300 chars):\n{obs2[:300]}\n")
        print("=== Test PASSED ===")
