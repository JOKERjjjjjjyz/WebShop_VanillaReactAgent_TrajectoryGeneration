#!/bin/bash
# ============================================================
# Start WebShop Environment Server for PARG_WebStore
# ============================================================
# Run this script to launch the WebShop env as an HTTP server
# so the agent (Python 3.10 / agent conda env) can call into it.
#
# Usage:
#   ./start_webshop_server.sh            # port 6001, 1000 products
#   ./start_webshop_server.sh 6001 1000  # explicit port and num_products
#
# Then in your agent code, use:
#   from runner.webshop_client import WebShopClient
#   env = WebShopClient("http://localhost:6001")
# ============================================================

PORT=${1:-6001}
NUM_PRODUCTS=${2:-1000}

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export WEBSHOP_PATH=/data1/yanze/WebShop
WEBSHOP_PYTHON=/data1/yanze/miniconda3/envs/webshop/bin/python

echo "[WebShop Server] Starting on port $PORT with $NUM_PRODUCTS products..."
echo "[WebShop Server] JAVA_HOME=$JAVA_HOME"
echo "[WebShop Server] Python=$WEBSHOP_PYTHON"
echo ""

cd "$WEBSHOP_PATH"
exec "$WEBSHOP_PYTHON" /data1/yanze/PARG_WebStore/runner/webshop_env_wrapper.py \
    --serve \
    --port "$PORT" \
    --num_products "$NUM_PRODUCTS"
