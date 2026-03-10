# Environment Setup: WebShop for PARG_WebStore

## Overview

PARG_WebStore 使用 [princeton-nlp/WebShop](https://github.com/princeton-nlp/WebShop) 作为 benchmark 环境，
通过 text 模式（无需浏览器）与 agent 交互。

---

## Architecture

由于 WebShop 依赖旧版 Python 包（numpy 1.22, gym 0.26），与 PARG 的 agent 训练环境（Python 3.10）存在冲突，
采用**两个独立 conda 环境 + HTTP server** 的架构：

```
┌──────────────────────────────────────┐   HTTP (localhost:6001)
│  agent env (Python 3.10)             │◄──────────────────────►  ┌──────────────────────────────┐
│  /data1/yanze/miniconda3/envs/agent  │                          │  webshop env (Python 3.8)    │
│                                      │                          │  /data1/yanze/miniconda3/    │
│  - PARG model training               │                          │    envs/webshop              │
│  - ReAct agent logic                 │                          │                              │
│  - TrajectoryLogger                  │                          │  - WebAgentTextEnv           │
│  - GraphBuilder                      │                          │  - Lucene search index       │
│  runner/webshop_client.py            │                          │  runner/webshop_env_wrapper  │
└──────────────────────────────────────┘                          └──────────────────────────────┘
```

**注**：如果不需要训练（只做 rollout），也可以直接在 webshop env 里 import PARG 模块。

---

## 目录结构

```
/data1/yanze/
├── WebShop/                         # WebShop 源码（princeton-nlp/WebShop）
│   ├── web_agent_site/              # 环境代码
│   ├── search_engine/
│   │   ├── indexes_1k/              # Lucene 索引（1000 products）
│   │   └── resources_1k/           # 索引原始文件
│   └── data/
│       ├── items_shuffle_1000.json  # 产品数据（1000 条）
│       ├── items_ins_v2_1000.json   # 产品属性
│       └── items_human_ins.json     # 任务指令
├── PARG_WebStore/
│   ├── runner/
│   │   ├── webshop_env_wrapper.py   # WebShop env + Flask server
│   │   ├── webshop_client.py        # HTTP client（给 agent env 用）
│   │   ├── agentbench_wrapper.py    # ReActWebShopAgent
│   │   └── trajectory_logger.py    # 轨迹日志
│   └── scripts/
│       └── start_webshop_server.sh  # 一键启动 server
└── miniconda3/
    ├── envs/agent/                  # Python 3.10，PARG 训练
    └── envs/webshop/                # Python 3.8，WebShop env
```

---

## 快速使用

### 方法 A：Server 模式（推荐，跨环境）

**Terminal 1：启动 WebShop server**
```bash
bash /data1/yanze/PARG_WebStore/scripts/start_webshop_server.sh
# 或者明确指定
bash /data1/yanze/PARG_WebStore/scripts/start_webshop_server.sh 6001 1000
```

**Terminal 2：在 agent env 里跑 agent**
```python
# 在 agent 环境 (Python 3.10) 里
from runner.webshop_client import WebShopClient
from runner.agentbench_wrapper import ReActWebShopAgent
from runner.trajectory_logger import TrajectoryLogger

env = WebShopClient("http://localhost:6001")
logger = TrajectoryLogger("logs/raw/webshop")
agent = ReActWebShopAgent(llm_client=..., logger=logger)

obs = env.reset(seed=42)
success = agent.run_episode(env, task_id=env.task_id, instruction=obs)
```

### 方法 B：直接模式（webshop env 里全跑）

```bash
# 在 webshop env 里直接运行整个 rollout
cd /data1/yanze/WebShop
JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
WEBSHOP_PATH=/data1/yanze/WebShop \
/data1/yanze/miniconda3/envs/webshop/bin/python \
    /data1/yanze/PARG_WebStore/runner/webshop_env_wrapper.py --test
```

---

## 环境变量

| 变量 | 值 | 说明 |
|---|---|---|
| `JAVA_HOME` | `/usr/lib/jvm/java-11-openjdk-amd64` | Lucene 搜索引擎需要 Java 11 |
| `WEBSHOP_PATH` | `/data1/yanze/WebShop` | WebShop 源码路径 |
| `WEBSHOP_PYTHON` | `/data1/yanze/miniconda3/envs/webshop/bin/python` | webshop 专用 Python |

---

## WebShop Action 格式

```
search[<query>]           # 搜索商品
click[<product_id>]       # 点击产品（如 click[B09QQWRW2M]）
click[<option>]           # 选择规格（如 click[large], click[blue]）
click[buy now]            # 购买（终止动作）
click[back to search]     # 返回搜索页
click[next >]             # 翻页
click[< prev]             # 上一页
```

---

## 数据规模

| 模式 | 产品数 | goals 数 |
|------|--------|----------|
| small (当前) | 1,000 | 6,910 |
| all | 1,180,000+ | 12,087 |

当前用 `num_products=1000` 运行，适合快速 debug 和 rollout pipeline 验证。
确认 pipeline 跑通后可改为 all（需要重新 `./setup.sh -d all` 和重建索引）。

---

## 安装验证

```bash
# 验证 env 能否正常运行
cd /data1/yanze/WebShop
JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
WEBSHOP_PATH=/data1/yanze/WebShop \
/data1/yanze/miniconda3/envs/webshop/bin/python \
    /data1/yanze/PARG_WebStore/runner/webshop_env_wrapper.py --test
```

期望输出：
- `Products loaded.`
- `Loaded 6910 goals.`
- `=== Test PASSED ===`

---

## 已知问题

1. **gym deprecation warning**：`gym==0.26.2` 会提示升级到 gymnasium，不影响运行，忽略即可。
2. **Lucene JVM warning**：`sun.reflect.Reflection.getCallerClass is not supported`，Java 11 兼容性 warning，忽略。
3. **num_products 加载较慢**（all 模式）：1.18M 产品加载需要几十秒，建议开发时用 1000。

---

## 后续扩展：全量数据

如果想跑 all data，在 webshop env 里：
```bash
cd /data1/yanze/WebShop
# 下载全量数据
./setup.sh -d all     # 在 webshop env 里

# 然后修改 web_agent_site/utils.py 使用全量文件：
# DEFAULT_ATTR_PATH = join(BASE_DIR, '../data/items_ins_v2.json')
# DEFAULT_FILE_PATH = join(BASE_DIR, '../data/items_shuffle.json')
```
