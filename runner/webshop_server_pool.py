"""
Multi-Session WebShop HTTP Server
=================================
Supports concurrent sessions by sharing a single expensive SimServer
across multiple WebAgentTextEnv instances.

Usage:
    # Start with webshop conda env (Python 3.8):
    conda activate webshop
    python runner/webshop_server_pool.py --port 6001 --num_products 1000

API:
    POST /create_session  {"task_idx": int|null}
        -> {"session_id": str, "obs": str, "instruction_text": str}

    POST /step  {"session_id": str, "action": str}
        -> {"obs": str, "reward": float, "done": bool, "info": dict}

    POST /close_session  {"session_id": str}
        -> {"status": "ok"}

    GET /health
        -> {"status": "ok", "active_sessions": int}
"""

import sys
import os
import uuid
import time
import threading
import logging

WEBSHOP_PATH = os.environ.get("WEBSHOP_PATH", "/data1/yanze/WebShop")
if WEBSHOP_PATH not in sys.path:
    sys.path.insert(0, WEBSHOP_PATH)

os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-11-openjdk-amd64")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebShopSessionPool:
    """Manages multiple WebAgentTextEnv instances sharing one SimServer."""

    def __init__(self, num_products=1000, session_ttl=600):
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

        logger.info("Loading shared SimServer (this may take a while)...")
        # Create one env to initialize the shared server
        self._template_env = WebAgentTextEnv(
            observation_mode="text",
            num_products=num_products,
        )
        self._shared_server = self._template_env.server
        self._num_products = num_products
        self._session_ttl = session_ttl

        # session_id -> {"env": WebAgentTextEnv, "created_at": float}
        self._sessions = {}
        self._lock = threading.Lock()

        # Start TTL cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

        logger.info(f"WebShopSessionPool ready. num_products={num_products}, ttl={session_ttl}s")

    def create_session(self, task_idx=None):
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

        session_id = str(uuid.uuid4())

        env = WebAgentTextEnv(
            observation_mode="text",
            num_products=self._num_products,
            server=self._shared_server,
        )

        if task_idx is not None:
            obs = env.reset(session=task_idx)
        else:
            obs = env.reset()

        # reset() returns (obs, info) tuple
        if isinstance(obs, tuple):
            obs = obs[0]

        instruction_text = env.instruction_text or ""

        with self._lock:
            self._sessions[session_id] = {
                "env": env,
                "created_at": time.time(),
            }

        logger.info(f"Created session {session_id[:8]}... task_idx={task_idx}")
        return session_id, obs, instruction_text

    def step(self, session_id, action):
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session {session_id} not found")
            env = session["env"]

        result = env.step(action)

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result

        if isinstance(obs, tuple):
            obs = obs[0]

        # Auto-cleanup on done
        if done:
            self._remove_session(session_id)

        return obs, float(reward), bool(done), info or {}

    def close_session(self, session_id):
        self._remove_session(session_id)

    def _remove_session(self, session_id):
        with self._lock:
            self._sessions.pop(session_id, None)

    def _cleanup_loop(self):
        while True:
            time.sleep(60)
            now = time.time()
            expired = []
            with self._lock:
                for sid, info in self._sessions.items():
                    if now - info["created_at"] > self._session_ttl:
                        expired.append(sid)
                for sid in expired:
                    self._sessions.pop(sid, None)
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired sessions")

    @property
    def active_session_count(self):
        with self._lock:
            return len(self._sessions)


def run_server(port=6001, num_products=1000, session_ttl=600):
    from flask import Flask, request, jsonify

    app = Flask(__name__)
    pool = WebShopSessionPool(num_products=num_products, session_ttl=session_ttl)

    @app.route("/create_session", methods=["POST"])
    def create_session():
        data = request.get_json(force=True, silent=True) or {}
        task_idx = data.get("task_idx")
        try:
            session_id, obs, instruction_text = pool.create_session(task_idx=task_idx)
            return jsonify({
                "session_id": session_id,
                "obs": obs,
                "instruction_text": instruction_text,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/step", methods=["POST"])
    def step():
        data = request.get_json(force=True, silent=True) or {}
        session_id = data.get("session_id", "")
        action = data.get("action", "")
        try:
            obs, reward, done, info = pool.step(session_id, action)
            return jsonify({
                "obs": obs,
                "reward": reward,
                "done": done,
                "info": info,
            })
        except KeyError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/close_session", methods=["POST"])
    def close_session():
        data = request.get_json(force=True, silent=True) or {}
        session_id = data.get("session_id", "")
        pool.close_session(session_id)
        return jsonify({"status": "ok"})

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "active_sessions": pool.active_session_count,
        })

    logger.info(f"Starting WebShop pool server on port {port}")
    app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-session WebShop HTTP server")
    parser.add_argument("--port", type=int, default=6001)
    parser.add_argument("--num_products", type=int, default=1000)
    parser.add_argument("--session_ttl", type=int, default=600, help="Session TTL in seconds")
    args = parser.parse_args()

    run_server(port=args.port, num_products=args.num_products, session_ttl=args.session_ttl)
