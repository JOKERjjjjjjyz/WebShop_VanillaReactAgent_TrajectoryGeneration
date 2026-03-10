# Claims & Scope

## Primary Claims (In Scope)
1. **Offline Training Pipeline**: We establish an end-to-end framework that constructs graph structures from multi-step tool-use trajectories using log semantics and explicit referential citations.
2. **Dual-Head Topology Model**: We propose a joint optimization methodology where empirical continuation failure rates predict step-level risk, and failure root-cause paths predict node-level anchors.
3. **Closed-loop Inference Control**: The dual risk/anchor heads successfully compose a zero-shot intervention policy (Gate, Verify, Replan, Backtrack) that substantially improves agent success rates compared to passive ReAct baselines.

## Non-Claims (Out of Scope)
1. This is not a generalized LLM reasoning model (e.g., we do not claim to beat GPT-4 on generic reasoning benchmarks, but rather on structural tool-use recovery).
2. We do not claim this as an RL method (like DPO or PPO), though it shares connections to value-function estimation and DAgger. We position it as pre-action risk filtering and graph-based rollback.
