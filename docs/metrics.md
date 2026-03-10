# Evaluation Metrics

## Trajectory-Level Metrics
- **Success Rate (SR)**: Binary indicator if the final submitted answer passes the environment's golden criteria.
- **Expected Failure Rate (EFR)**: Equivalent to $1 - SR$, used as the primary target for predicting risk $R_t$.
- **Cost**: Measured via a combined proxy of API tokens consumed and the total number of Tool calls invoked.

## Node-Level Metrics (Anchor Localization)
- **Hit@k**: For failed trajectories, the percentage of instances where the human-labeled (or LLM-labeled) root cause appears in the top-k highest probability items predicted by the Anchor Head.

## Control-Policy Metrics
- **Intervention Count**: Number of times the active Gate intercepted a high-risk action.
- **Recovery Success Rate**: The percentage of trajectories that initially triggered a Backtrack operation but eventually succeeded.
