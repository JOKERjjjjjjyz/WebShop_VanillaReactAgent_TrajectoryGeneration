# Problem Setup and Variables

## Problem Setting
We consider a single-agent interactive decision-making setting, specifically modeled around Web-based tasks (e.g., WebShop). The agent interacts with an environment over discrete time steps to complete an instruction.

## Agent Architecture
- **Framework**: ReAct (Reason + Act). At each step, the agent outputs a `THOUGHT`, optionally followed by an `ACT_TOOL` (action) or an `ANSWER` (final prediction).
- **Graph Extraction**: The sequence of interactions is parsed into a heterogeneity graph $G_T=(V_T, E_T)$.

## Variables
- $\tau$: A trajectory of length $T$.
- $G_t$: The prefix graph constructed up to step $t$.
- $v_i \in V_T$: A node in the graph, typed appropriately (`THOUGHT`, `ACT_TOOL`, `OBS_TOOL`, etc.).
- $e \in E_T$: An explicit edge capturing either temporal flow or citation-based dependency.
- $R_t \in [0, 1]$: Pre-action condition failure probability (Risk).
- $A_i \in [0, 1]$: Node-level root-cause attribution probability (Anchor).
