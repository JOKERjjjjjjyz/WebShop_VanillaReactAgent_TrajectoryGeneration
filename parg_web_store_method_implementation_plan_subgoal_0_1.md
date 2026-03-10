# Method Spec + Implementation Plan (Subgoal 0–1)

> Scope: single-agent, tool-using ReAct workflow on WebStore/WebShop-style benchmark.
> Goal: build an **offline** risk+anchor training pipeline from **real rollouts**, then plug into a dual-gate controller.

---

## 0. Project Method (Clear End-to-End Description)

### 0.1 Setting

- **Environment / Benchmark**: a multi-step decision-making benchmark with an interactive web/store interface.
- **Agent**: vanilla **ReAct** loop (Thought → Act → Observation), optionally with **Memory** operations.
- **Tools** (example; you will instantiate for WebStore):
  - `web_search(query)` / `web_browse(url)` / `web_click(selector)` / `web_extract(schema)`
  - optional: `memory_read(key/topk)` / `memory_write(key, value)`
- **Episode**: one task instance produces a trajectory \(\tau\) until termination (success/fail) or budget.

### 0.2 Trajectory as a Graph

Each run yields a dynamic heterogeneous graph \(G_T=(V_T,E_T)\):

- **Nodes \(V_T\)** (typed):
  - `THOUGHT` (reasoning text)
  - `ACT_TOOL` (tool name + args)
  - `OBS_TOOL` (tool output / extracted evidence)
  - `MEM_READ` / `MEM_WRITE` (optional)
  - `ANSWER` (final answer / final action)

- **Edges \(E_T\)** (typed):
  - **Temporal**: \(v_i \to v_{i+1}\)
  - **Tool linkage**: `ACT_TOOL(i) -> OBS_TOOL(i)`
  - **Evidence-use / reference**:
    - `OBS_TOOL(j) -> THOUGHT/ACT_TOOL(i)` if later step references that observation
    - `MEM_READ(j) -> THOUGHT/ACT_TOOL(i)` if later step uses retrieved memory
  - **Write provenance**:
    - `THOUGHT/OBS_TOOL(i) -> MEM_WRITE(i)`

> Implementation note: edges must be **deterministically extractable** from logs (see Subgoal 1).

### 0.3 Two Heads, Two Supervision Targets

#### (A) Risk head (Decision Risk)

- **Input**: current prefix graph \(G_t\) (or candidate \(G_t^{cand}\)).
- **Output**: \(R_t \in [0,1]\), interpreted as
  \[
  R_t \approx \Pr(\mathrm{fail}\mid G_t, \pi)\quad\text{or}\quad\Pr(\mathrm{catastrophe}\mid G_t,\pi)
  \]
- **Usage**: gate the next action or trigger `VERIFY / REPLAN / BACKTRACK`.

#### (B) Anchor head (Root-cause localization)

- **Input**: (typically) a failure episode graph \(G_T\).
- **Output**: node-wise \(A_i \in [0,1]\) for all nodes \(v_i\), indicating “root-cause candidate”.
- **Usage**: decide **where to backtrack** or which evidence/memory chunk to quarantine.

### 0.4 Training Data from Real Rollouts (Offline)

1) **Collect rollouts** with behavior policy \(\mu\) (vanilla ReAct, high temperature for diversity).
2) **Parse logs** → graph \(G_T\), also store every prefix \(G_t\).
3) Build labels:
   - **Risk labels**: estimated conditional failure probability \(\hat R(G_t)\) via
     - **Empirical prefix conditioning** (bucket similar prefixes) and/or
     - **Sparse continuation** (only for key prefixes) for calibration.
   - **Anchor labels**: for failed runs only, annotate anchor nodes using a deterministic guideline + (optional) LLM-aided selection among candidates.
4) Train model with two heads:
   - \(\mathcal{L}=\mathcal{L}_{risk}+\lambda\,\mathcal{L}_{anch}\)

### 0.5 Dual-Gate Controller (Deployment)

Given current prefix \(G_t\):

1) Compute \(R_t\) via risk head.
2) If \(R_t\le \tau_r\): **PASS** (execute next ReAct action).
3) Else (high risk):
   - run a lightweight **verifier gate** (e.g., consistency checks / extra retrieval)
   - if verifier clears: PASS
   - else: use anchor head to pick top-\(k\) anchor(s) and **BACKTRACK**:
     - drop/rollback contaminated memory/evidence
     - replan and continue.

---

## Subgoal 0 — Lock the Setup, Claims, and Protocol (Fine-grained Plan)

### 0.0 Deliverables

- `docs/setup.md` — formal problem setup + symbols
- `docs/claims.md` — scope/claims + non-claims
- `docs/metrics.md` — metrics definitions (success, cost, propagation)
- `docs/repro.md` — seeds, configs, environment pinning
- `schemas/trajectory.schema.json` — log schema
- `schemas/graph.schema.json` — graph schema

### 0.1 Decide and freeze the **benchmark contract**

1) **Task format**
   - fields: `task_id`, `instruction`, optional `initial_context`, `gold` (if applicable)
2) **Success criterion** (single boolean) `is_success(episode)`
   - define canonical parser for final state
3) **Budgets**
   - `max_steps`, `max_tool_calls`, `max_tokens` (if tracked)

Acceptance check:
- Same episode re-run with same seed yields same termination and success/fail (modulo model stochasticity, so store sampled outputs).

### 0.2 Freeze the **agent policy** \(\mu\) used for data collection

- ReAct prompt template (versioned)
- Temperature/top-p
- Tool set allowed + tool-call formatting rules
- Termination rules (when to answer)

Outputs:
- `prompts/react_v1.txt`
- `configs/collector_v1.yaml`

### 0.3 Define the **action / node taxonomy** (must be stable)

- Node types: `THOUGHT`, `ACT_TOOL`, `OBS_TOOL`, `MEM_READ`, `MEM_WRITE`, `ANSWER`
- For each type, define required fields:
  - e.g., `ACT_TOOL`: `tool_name`, `args_json`, `call_id`
  - `OBS_TOOL`: `call_id`, `raw_text`, `parsed_json` (optional)

Outputs:
- `schemas/trajectory.schema.json`

### 0.4 Define **edge extraction rules** (deterministic)

Edge types and how to detect:

1) Temporal edges: always
2) Tool linkage: by `call_id`
3) Reference edges (core): you must choose one deterministic mechanism:
   - Option A: **explicit citation ids** in agent outputs (preferred)
     - agent must refer to evidence as `[[obs:12]]`, `[[mem:3]]`, etc.
   - Option B: heuristic matching (fallback)
     - string overlap / url match / entity match (less defendable)

Outputs:
- `docs/graph_extraction.md`
- `graph/build_graph.py`

Acceptance check:
- 100% of logs parse into graphs without manual intervention.

### 0.5 Define **labels** precisely (before building models)

Risk label definition (choose primary + optional calibration):

- Primary: **empirical conditional failure rate** under \(\mu\)
  - \(\hat R(G_t)=\frac{\#fail\ \text{among similar prefixes}}{\#total}\)
- Calibration (optional): sparse continuation on selected prefixes.

Anchor label definition:
- for failed episodes only, anchor is the earliest node satisfying:
  - introduces incorrect premise OR accepts contaminated observation/memory
  - and lies on dependency path to final failure

Outputs:
- `docs/labels.md`

Acceptance check:
- A third party can label a sample episode using only the guideline.

### 0.6 Define baselines and ablations (to pre-empt Copilot/AgentHallu pressure)

Minimum list (document now, implement later):
- vanilla ReAct (no gates)
- risk-only gate (no anchor/backtrack)
- topology-only vs semantic-only
- Copilot-like single-head failure predictor (graph → fail)

Output:
- `docs/baselines.md`

---

## Subgoal 1 — Build the Rollout + Logging + Dataset Builder (Fine-grained Plan)

### 1.0 Deliverables

- `runner/` — benchmark runner + agent wrapper
- `tools/` — tool server/client + typed schemas
- `logs/raw/*.jsonl` — raw trajectories
- `datasets/` — risk/anchor datasets (prefix slices)
- `scripts/collect_rollouts.py` — data collection
- `scripts/build_datasets.py` — dataset build

### 1.1 Environment & tool interface (WebStore)

1) Define tool APIs (Python functions or HTTP endpoints):
   - `search(query) -> {results}`
   - `open(url) -> {html/text}`
   - `click(selector) -> {new_page}`
   - `extract(schema) -> {json}`
   - (optional) `memory_read(topk/key)` / `memory_write(key,val)`
2) Ensure tool outputs are **machine-readable**:
   - store `raw_text` + `parsed_json`
3) Add tool-level metadata:
   - latency, status, error codes

Acceptance check:
- Tools can be replayed deterministically from recorded args (for audit).

### 1.2 Agent wrapper (ReAct collector policy \(\mu\))

1) Implement a single step function:
   - input: current observation + scratchpad + memory snippets
   - output: `THOUGHT` + optional `ACT_TOOL` or `ANSWER`
2) Enforce tool-call format and explicit references:
   - if using citation-based edges: require `[[obs:i]]` / `[[mem:j]]`
3) Temperature schedule:
   - set high temp for plan diversity; optionally lower temp for tool arg formatting stability

Acceptance check:
- 100 episodes run without formatting crashes; tool calls validate against schema.

### 1.3 Unified logging (raw trajectory JSONL)

For each episode, write a JSON object:
- episode metadata: `task_id`, `seed`, `policy_version`, `toolset_version`
- steps: list of step records

Each step record must include:
- `step_id` (monotonic)
- `node_type`
- payload fields (per schema)
- `call_id` for tool steps
- explicit `ref_ids` used at this step (if citation-based)

Output file naming:
- `logs/raw/{benchmark}/{policy_version}/{date}/episodes.jsonl`

Acceptance check:
- A standalone parser can reconstruct the episode without running the agent.

### 1.4 Graph builder (G construction)

1) Parse each episode log into nodes \(V\) with unique node ids.
2) Build edges:
   - temporal
   - tool linkage (call_id)
   - reference edges from explicit `ref_ids`
3) Compute per-node derived features:
   - depth, out-degree, in-degree
   - tool type one-hot
   - evidence count referenced

Outputs:
- `graphs/{episode_id}.pt` (or json) containing nodes, edges, node_features
- `graphs/index.csv` mapping episode_id → task_id, success, length

Acceptance check:
- No missing edge types; graphs pass schema validation.

### 1.5 Dataset slicing for training

Goal: create samples for training heads.

**Risk dataset (prefix slices)**
- For each episode and each selected prefix index \(t\):
  - input: \(G_t\) (subgraph induced by first t nodes)
  - label: \(\hat R(G_t)\)

Prefix selection strategies:
- `ALL`: all t (expensive)
- `KEY`: only key t where node_type in {MEM_READ, MEM_WRITE, OBS_TOOL}

**Anchor dataset (episode-level)**
- For each failed episode:
  - input: \(G_T\)
  - label: anchor indices (one-hot or top-k)

Outputs:
- `datasets/risk_v1.jsonl` (or pt)
- `datasets/anchor_v1.jsonl`

Acceptance check:
- Dataset size and class balance reported (fail rate, anchors per episode, etc.).

### 1.6 Risk labeler implementation (two-tier)

Tier 1: empirical prefix conditioning
1) Build prefix signatures `sig(G_t)`:
   - tool last type + referenced evidence ids hash + degree stats bucket
2) Group samples by `(task_id, t, sig)` or by `(sig)` depending on sparsity
3) For each group, compute fail rate as \(\hat R\)

Tier 2 (optional): sparse continuation calibration
1) Choose M key prefixes per episode by heuristics:
   - high out-degree nodes
   - memory write nodes
2) Re-run from saved state at t with different seeds to get K' suffix outcomes
3) Calibrate / validate Tier 1 labels

Outputs:
- `datasets/risk_v1_with_labels.jsonl`
- `reports/risk_label_quality.md` (sparsity, bucket sizes, calibration correlation)

Acceptance check:
- Label quality: bucket sizes not too small; calibration agrees directionally.

### 1.7 Anchor labeler implementation

1) Candidate generation (rule-based):
   - find dependency path(s) to terminal failure node
   - shortlist earliest nodes on path with types {OBS_TOOL, MEM_READ, MEM_WRITE, THOUGHT}
2) LLM-aided selection (optional but recommended):
   - provide candidates + episode summary + failure description
   - ask for top-1/top-k anchor ids + short justification
3) Consistency check:
   - dual-prompt agreement or spot-checking

Outputs:
- `datasets/anchor_v1_with_labels.jsonl`
- `reports/anchor_label_agreement.md`

Acceptance check:
- Non-triviality: anchors are not always first tool call; distribution looks plausible.

### 1.8 Baseline exports (for immediate sanity)

- `reports/baseline_react.md`: success, avg steps, avg tool calls
- `reports/failure_taxonomy.md`: basic breakdown by failure type (optional)

---

## Immediate Next Actions (what to implement first in antigravity)

1) **Finalize schemas**: `trajectory.schema.json` + `graph.schema.json`
2) **Collect 50–100 episodes** with stable logging
3) **Build graphs** and validate edge extraction
4) **Produce baseline report** (success/cost)
5) **Implement Tier-1 risk labeler** (prefix conditioning) and print label statistics

Once these pass, you can start training heads without fear of redoing the pipeline.

