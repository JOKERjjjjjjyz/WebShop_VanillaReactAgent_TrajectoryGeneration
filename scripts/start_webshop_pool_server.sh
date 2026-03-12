#!/bin/bash
# ============================================================
# Start Multi-Session WebShop Pool Server
# ============================================================
# Supports concurrent sessions for OpenRLHF async RL training.
# Uses conda env "webshop" (Python 3.8).
#
# Usage:
#   ./start_webshop_pool_server.sh             # port 6001, 1000 products
#   ./start_webshop_pool_server.sh 6001 1000   # explicit port and num_products
# ============================================================

PORT=${1:-6001}
NUM_PRODUCTS=${2:-1000}

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export WEBSHOP_PATH=/data1/yanze/WebShop
WEBSHOP_PYTHON=/data1/yanze/miniconda3/envs/webshop/bin/python

echo "[WebShop Pool Server] Starting on port $PORT with $NUM_PRODUCTS products..."
echo "[WebShop Pool Server] JAVA_HOME=$JAVA_HOME"
echo "[WebShop Pool Server] Python=$WEBSHOP_PYTHON"
echo ""

cd "$WEBSHOP_PATH"
exec "$WEBSHOP_PYTHON" /data1/yanze/PARG_WebStore/runner/webshop_server_pool.py \
    --port "$PORT" \
    --num_products "$NUM_PRODUCTS"
