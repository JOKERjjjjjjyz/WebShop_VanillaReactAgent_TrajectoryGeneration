import os
import json
import time
from typing import Dict, Any, List

class TrajectoryLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.current_episode = None
    
    def start_episode(self, episode_id: str, task_id: str, seed: int, policy_version: str = "v1", toolset_version: str = "v1"):
        self.current_episode = {
            "episode_id": episode_id,
            "task_id": task_id,
            "seed": seed,
            "policy_version": policy_version,
            "toolset_version": toolset_version,
            "metadata": {},
            "steps": [],
            "outcome": {}
        }
        self.step_counter = 0

    def log_node(self, node_type: str, payload: Dict[str, Any], call_id: str = None, ref_ids: List[int] = None):
        """
        Logs a single node (THOUGHT, ACT_TOOL, OBS_TOOL, etc.).
        Returns the assigned step_id.
        """
        if not self.current_episode:
            raise ValueError("No active episode. Call start_episode() first.")
        
        step_id = self.step_counter
        self.step_counter += 1
        
        node = {
            "step_id": step_id,
            "node_type": node_type,
            "payload": payload,
            "timestamp": time.time()
        }
        if call_id:
            node["call_id"] = call_id
        if ref_ids is not None:
            node["ref_ids"] = ref_ids
            
        self.current_episode["steps"].append(node)
        return step_id

    def end_episode(self, is_success: bool, total_cost: float, termination_reason: str):
        if not self.current_episode:
            raise ValueError("No active episode.")
        
        self.current_episode["outcome"] = {
            "is_success": is_success,
            "total_cost": total_cost,
            "termination_reason": termination_reason
        }
        
        date_str = time.strftime("%Y-%m-%d")
        log_file = os.path.join(self.log_dir, f"{date_str}_episodes.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.current_episode) + "\n")
            
        self.current_episode = None
