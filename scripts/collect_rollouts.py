import argparse
from tqdm import tqdm
from runner.trajectory_logger import TrajectoryLogger
from runner.agentbench_wrapper import ReActWebShopAgent
from runner.local_llm_client import LocalHuggingFaceClient
from tqdm import tqdm
from runner.trajectory_logger import TrajectoryLogger
from runner.agentbench_wrapper import ReActWebShopAgent

# Note: In a real environment, WebAgentTextEnv is imported from WebShop
try:
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
except ImportError:
    print("Warning: WebShop environment not found. Please install principle-nlp/WebShop.")
    WebAgentTextEnv = None

class DummyLLM:
    """A dummy LLM for testing the pipeline when no API key is provided."""
    def __init__(self):
        self.step = 0
    def chat(self, history):
        self.step += 1
        if self.step == 1:
            return "Thought: I need to search for an apple.\nAction: search[apple]"
        return "Thought: I found it. [[obs:1]]\nAction: click[buy]"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--log_dir", type=str, default="logs/raw/webshop")
    args = parser.parse_args()

    logger = TrajectoryLogger(log_dir=args.log_dir)
    
    # Initialize the local Qwen 0.5B model instead of an API
    llm = LocalHuggingFaceClient(model_name_or_path="Qwen/Qwen1.5-0.5B-Chat") 
    agent = ReActWebShopAgent(llm_client=llm, logger=logger, policy_version="react_v1")

    if WebAgentTextEnv is None:
        print("Cannot run rollouts without WebShop environment. Exiting.")
        return

    # Assume WebShop server is running locally on port 3000
    print(f"Collecting {args.num_episodes} rollouts...")
    success_count = 0
    
    for i in tqdm(range(args.num_episodes)):
        env = WebAgentTextEnv(observation_mode="text", human_goals=True)
        # Assuming environment has an instruction property after reset
        # For WebShop, instruction is often part of the initial observation
        
        task_id = f"webshop_task_{i}"
        
        try:
            success = agent.run_episode(env=env, task_id=task_id, instruction="Buy me a good product", max_steps=15)
            if success:
                success_count += 1
        except Exception as e:
            print(f"Episode {i} failed with error: {e}")
            
    print(f"Finished collection. Success rate: {success_count}/{args.num_episodes}")

if __name__ == "__main__":
    main()
