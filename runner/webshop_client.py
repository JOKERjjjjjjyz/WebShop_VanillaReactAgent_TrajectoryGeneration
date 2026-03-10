#!/usr/bin/env python3
"""
WebShop HTTP Client
===================
When running in cross-env mode (PARG agent env = Python 3.10 
vs WebShop env = Python 3.8), use this client to talk to the 
WebShop HTTP server started via:

    WEBSHOP_PATH=/data1/yanze/WebShop \\
    JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \\
    /data1/yanze/miniconda3/envs/webshop/bin/python \\
    /data1/yanze/PARG_WebStore/runner/webshop_env_wrapper.py \\
    --serve --port 6001

This client can be used directly from the agent env (Python 3.10).
It exposes the same interface as WebShopEnv (reset/step).
"""

import requests


class WebShopClient:
    """
    Drop-in replacement for WebShopEnv when running in cross-env mode.
    Talks to a WebShopEnv HTTP server.

    Compatible with agentbench_wrapper.py (env.step / env.reset).
    """

    def __init__(self, base_url: str = "http://localhost:6001"):
        self.base_url = base_url.rstrip("/")
        self.seed = 42
        self.task_id = None

    def reset(self, task_idx: int = None, seed: int = None) -> str:
        payload = {}
        if task_idx is not None:
            payload["task_idx"] = task_idx
        if seed is not None:
            payload["seed"] = seed

        resp = requests.post(f"{self.base_url}/reset", json=payload)
        resp.raise_for_status()
        data = resp.json()
        self.seed = data.get("seed", 42)
        self.task_id = data.get("task_id")
        return data["obs"]

    def step(self, action: str):
        resp = requests.post(f"{self.base_url}/step", json={"action": action})
        resp.raise_for_status()
        data = resp.json()
        return data["obs"], data["reward"], data["done"], data.get("info", {})

    def get_available_actions(self):
        resp = requests.get(f"{self.base_url}/actions")
        resp.raise_for_status()
        return resp.json()


if __name__ == "__main__":
    # Quick smoke test — requires server to be running
    client = WebShopClient()
    obs = client.reset(seed=42)
    print("=== Client Test ===")
    print("Obs:", obs[:300])
    obs2, r, done, info = client.step("search[water bottle]")
    print("After search:", obs2[:200])
    print("reward:", r, "done:", done)
