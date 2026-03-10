# Baseline ReAct Report (V2)

**Generated**: 2026-03-10 21:18:32  
**Model**: Qwen3.5-4B (vLLM 0.17.0, `bfloat16`, single GPU)  
**Environment**: WebShop text mode, `num_products=1000`  
**Policy**: vanilla ReAct, `policy_version=react_v1`  
**Tasks sampled**: 500 / 6910 total goals  
**Max steps / episode**: 1000  
**Parallel workers**: 4  

---

## Results Summary

| Metric | Value |
|--------|-------|
| **Full success** (reward = 1.0) | 161 / 500 = **32.2%** |
| **Partial success** (0 < reward < 1) | 138 / 500 = **27.6%** |
| **Failure** (reward = 0) | 201 / 500 = **40.2%** |
| **Average reward** | **0.4857** |
| **Average steps** | 31.32 |
| **Average format errors/episode** | 1.32 |

## Token Usage

| Metric | Value |
|--------|-------|
| **Average final context tokens** | 23374 |
| **Min context tokens** | 3230 |
| **Max context tokens** | 59656 |

## Failure-Point Analysis

Total failure points detected across all episodes: **13224**  
Average failure points per episode: **26.45**  

| Metric | Value |
|--------|-------|
| Episodes with ≥1 failure point | 479 (95.8%) |

## Termination Breakdown

| Reason | Count |
|--------|-------|
| `done` | 301 (60.2%) |
| `format_failure` | 199 (39.8%) |

## Reward Distribution

| Reward bucket | Count | % |
|---|---|---|
| `0.0` | 201 | 40.2% |
| `(0, 0.5)` | 45 | 9.0% |
| `[0.5, 1.0)` | 93 | 18.6% |
| `1.0` | 161 | 32.2% |

## Files

| File | Description |
|------|-------------|
| `logs/raw/webshop/react_v1/<date>_baseline.jsonl` | Full trajectories with token counts + failure annotations |
| `reports/baseline_react.jsonl` | Summary stats per episode (no trajectory) |
| `reports/baseline_react.md` | This report |
