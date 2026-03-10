整体以“先 WebStore 做出有效”为主线，同时正面处理 AgentHallu 和 Trajectory Graph Copilot 的威胁。

---

# 研究子目标与实现计划

## Subgoal 0：锁定问题设定与“卖点边界”

**目的**：把你的贡献写成清晰可 defend 的“问题—方法—评估”闭环，避免被拉去比 RL/uncertainty 或被质疑定义不清。

**产出物**

* `setup.md`：任务、工具、日志 schema、成功判定、成本预算
* `claims.md`：主张范围（方法而非通用模型）、不做/可选做的扩展（DAgger-style）

**实现要点**

1. 固定 single-agent + ReAct action schema（Thought/Act/Obs + Memory ops），列出 tool 列表与参数格式
2. 定义图 (G_t=(V_t,E_t))：节点类型与边类型（time edge、toolcall→obs、memory_read→use、obs→use 等），明确抽取规则
3. 定义终局指标（EFR）与成本（token/tool calls/latency proxy）

**验收**

* 任意一条日志能 deterministically 解析成图（可重现）
* 任何人按文档能复现实验环境与成功判定

---

## Subgoal 1：建立“真实轨迹数据管线”与最小可跑基线

**目的**：先有一个完全可跑的 end-to-end baseline（vanilla ReAct）与可规模化采样的 rollout pipeline，后面所有标签、训练、对比都依赖它。

**产出物**

* `runner/`：benchmark runner + agent wrapper + tool server（或环境接口）
* `logs/`：统一 JSONL 轨迹（包含 node_id、node_type、tool、args、obs、memory_refs、timestamp、parent_refs）
* `baseline_react_results.csv`：每个 task 的 success、steps、cost

**实现步骤**

1. 实现可重复的 rollout：seed、temperature、max_steps、tool timeouts
2. 统一日志：每一步必须记录 action 与 observation；memory read/write 必须显式记录引用的 chunk ids
3. 采样策略：每个 task 跑 K 次（高温），获得分布

**验收**

* 能在 WebStore 上跑出稳定的 baseline success（波动可接受但可复现）
* 日志可用于后续“图构建 + 标签构建”而不返工

---

## Subgoal 2：Risk label 的可学监督：从“手工曲线”升级为“条件失败概率”

**目的**：绕开最容易被打穿的点：risk 标签不能是你人为规定的随时间增长曲线，必须来自经验条件失败率或 continuation 估计。

**产出物**

* `risk_dataset.jsonl`：样本为 (prefix_graph, risk_target, metadata)
* `risk_labeler.py`：从 rollouts 生成 risk label 的脚本
* `ablation_stepindex_baseline.py`：只用 step index/length 预测 risk 的弱基线

**实现路线（推荐优先级）**
**路线 A：prefix 分桶经验估计（最省资源，先做）**

1. 为每个 prefix (G_t) 计算“状态签名” (s(G_t))（例如：最近 tool 类型 + 关键 memory/evidence id 集合摘要 + DAG 结构统计）
2. 对同一 task 的 K 条轨迹，在同一 t 下按签名聚类
3. risk target = 该簇内失败比例

**路线 B：sparse continuation（补强论证）**

1. 每条轨迹选 M 个关键 prefix（memory read/write、关键 obs、hub 节点）
2. 从这些 (G_t) 做 K’ 次 suffix rollout，risk target = 失败比例
3. 用于校准/验证路线 A 的标签质量

**验收**

* “只用 step index”预测 risk 的 AUC/ECE 明显弱于你的模型（否则说明标签退化成时间启发式）
* risk target 与关键结构事件有相关性（例如某些 memory contamination 后风险陡升）

---

## Subgoal 3：Anchor label 的规范化：对齐 AgentHallu 式 attribution，但保留你的差异点

**目的**：正面处理 AgentHallu 的威胁：它把 attribution 标注成 benchmark 任务。你的策略应是“对齐定义 + 强调你是用于控制的 anchor（for recovery），不是仅事后解释”。

**产出物**

* `anchor_guideline.md`：操作化定义（可复现）
* `anchor_dataset.jsonl`：失败轨迹的 (full_graph, anchor_targets, evidence)
* `anchor_labeler.py`：规则 + LLM 辅助的标注 pipeline + 抽样复核脚本

**实现步骤**

1. Anchor 的可复现定义（必须写成规则）：

   * “最早引入错误前提且被后续依赖引用、并在 counterfactual 移除后显著降低失败概率”的节点（规则版可近似：依赖回溯 + 最早污染点）
2. 标注流程：

   * Rule-based 初筛候选 anchors（沿依赖边回溯）
   * LLM 在候选集合内选择 top-1/top-k，并给出短 justification（用于审稿可解释性）
3. 一致性验证：

   * 双 prompt 或双模型一致率
   * 抽样人工复核（哪怕 50 条）

**验收**

* anchor head 的 hit@k 在“近似症状节点”干扰下仍显著高于随机/启发式（例如“第一个 tool call”）
* 你能在论文里解释 anchor 与 root-cause 的区别：anchor 用于 backtracking/recovery 的最小修复点

---

## Subgoal 4：模型实现与关键对标：如何顶住 Trajectory Graph Copilot 的压力

Trajectory Graph Copilot 的威胁点是：它也做“pre-action error diagnosis + GNN”。你的计划必须在实现上做到：

* 有**强 baseline 对标**（否则审稿人会说你重复）
* 有**清晰差异**（你有 dual-head + control policy + recovery gains）

**产出物**

* `models/`：你的 dual-stream + dual-head 模型（semantic Transformer + topology GNN + fusion）
* `baseline_models/`：

  * B1: 纯 Transformer（只看序列，不看图）
  * B2: 纯 GNN（只看图，不看语义或简化语义）
  * B3: “Copilot-like” 单头 pre-action failure predictor（graph + classifier，预测 fail）
* `training/`：统一 trainer + ablation flags

**实现步骤**

1. 先把输入输出对齐你的控制需求：

   * risk head：tip-level (R(G_t)) 或 (Q(G_{t-1},a_t))（建议先做 tip-level，定义最干净）
   * anchor head：node-level (A_i)
2. 做“三件必须做的 ablation”：

   * 去掉 topology（只剩 semantic）
   * 去掉 semantic（只剩 topology）
   * 去掉 anchor/recovery（只做 risk gate）
3. Copilot-like baseline：

   * 同样的图输入，但只预测 “该步是否将导致最终失败” 的单头分类（不做 anchor，不做 recovery policy）

**验收**

* 你的方法在 EFR vs cost 的 Pareto 上优于 Copilot-like baseline（这是你对标它的关键）
* anchor + recovery 带来的提升是独立增益（不是 risk gate 的附属）

---

## Subgoal 5：Dual-gate 控制器落地与主实验闭环

**目的**：把你的方法从“预测”变成“控制”。否则很容易被认为只是诊断工具。

**产出物**

* `controller.py`：PASS / VERIFY / REPLAN / BACKTRACK / FIREWALL 的策略
* `eval_end2end.py`：同预算下的 success、cost、recovery stats
* 主表格与消融表格的数据导出脚本

**实现步骤**

1. Gate 触发规则明确化（可写成 deterministic policy）：

   * risk > τ_r → verify 或 replan
   * verify fail → backtrack 到 anchor top-1/top-k
2. Backtrack 的语义要定义清楚：回退不是“抹掉现实”，而是

   * 丢弃污染 memory/evidence
   * 重新检索/重新规划
3. 评估必须报告：

   * success(EFR)、成本、触发次数、回退深度、恢复成功率

**验收**

* one-shot offline 训练后，controller 在 WebStore 上有显著提升（并且 cost 没爆炸）
* recovery 成为实证亮点：不仅是“更谨慎”，而是“能救回来”

---

## Subgoal 6：DAgger-style 扩展（作为增强，而不是主线）

**目的**：你说得很对：主线卖 one-shot offline；DAgger-style 是增强章节，展示分布漂移缓解。

**产出物**

* `iterative_collect.py`：用 gated policy 再采样新轨迹并入训练集
* `round_results.csv`：round0/round1/round2 的性能曲线

**实现步骤**

1. Round 0：在 (\mu)=vanilla ReAct 日志上训
2. 部署 gating policy 得到 (\pi_1)，采集数据
3. 合并数据再训得到 (\theta_1)
4. 最多做 1–2 轮（足够出趋势，别把项目拖进 RL 深坑）

**验收**

* round1 相比 round0 明显更好（特别是 OOD 状态下的风险校准/鲁棒性）
* 你能在 paper 里用“dataset aggregation reduces distribution shift empirically”一句话带过，不引战 RL

---

# 时间与优先级建议（按“最少返工”排序）

1. Subgoal 1（runner+logging）
2. Subgoal 2（risk label：先分桶，后 sparse continuation 校准）
3. Subgoal 5（controller 闭环 + one-shot 主结果）
4. Subgoal 3（anchor 标注 + anchor head）
5. Subgoal 4（对标 Trajectory Graph Copilot 的 baseline 与 ablations）
6. Subgoal 6（DAgger-style 作为增强）

---

# 针对两篇“威胁论文”的应对策略（一句话版）

* **AgentHallu**：它证明 attribution 是硬任务；你要做的是“用于恢复控制的 anchor”，强调 **control gain**（回退后成功率提升）而不仅是 attribution accuracy。
* **Trajectory Graph Copilot**：它做预警；你要证明你做的是 **预警 + 定位 + 干预闭环**，并用 EFR–cost Pareto + recovery stats 压过“只预警”的单头方法。