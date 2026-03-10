import torch
import torch.nn as nn
from p1_model import PARGModel
from vectorize_trajectory import TrajectoryVectorizer

class DualGateController:
    def __init__(self, model_path=None, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = PARGModel().to(self.device)
        self.model.eval()
        
        # Load weights if path is provided
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
        self.vectorizer = TrajectoryVectorizer(device=self.device)
        
        # Gate Thresholds
        self.tau_v = 0.40 # Threshold for Semantic Retry (Gate 1)
        self.tau_r = 0.75 # Threshold for Structural Quarantine (Gate 2)

    def evaluate_state(self, current_trajectory):
        """
        Evaluates the current online trajectory and determines the macro-policy execution.
        
        Args:
            current_trajectory (list of dicts): The sequence of trajectory steps observed so far.
                ex: [{'action': 'thought', 'content': '...'}, {'action': 'retrieve_memory', 'content': '...'}]
                
        Returns:
            dict containing:
                - status: 'PASS', 'RETRY_STEP', or 'BACKTRACK'
                - anchor_node: The index of the historical contamination point (if BACKTRACK)
                - risk_score: The continuous risk probability R_t
        """
        
        if not current_trajectory:
            return {'status': 'PASS', 'risk_score': 0.0}
            
        t = len(current_trajectory)
        
        # 1. Vectorization
        with torch.no_grad():
            seq_x = self.vectorizer.vectorize_trajectory(current_trajectory).unsqueeze(0).to(self.device) # (1, t, 128)
            seq_lens = torch.tensor([t], dtype=torch.long, device=self.device)
            
            graph_x = seq_x.squeeze(0) # (t, 128)
            
            # 2. Build edge index (linear chain for online inference)
            edges_src = []
            edges_tgt = []
            for i in range(t - 1):
                edges_src.append(i)
                edges_tgt.append(i + 1)
                
            if not edges_src:
                edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
            else:
                edge_index = torch.tensor([edges_src, edges_tgt], dtype=torch.long, device=self.device)
                
            context_x = torch.tensor([[float(t), 0.0, 0.0]], dtype=torch.float32, device=self.device)
            graph_batch = torch.zeros(t, dtype=torch.long, device=self.device)
            
            # 3. Model Forward Pass
            r_prop, a_nodes = self.model(seq_x, seq_lens, graph_x, edge_index, context_x, graph_batch)
            
            r_val = r_prop.item()
            a_probs = a_nodes.cpu().numpy()
            
        # 4. Dual-Gate Logic Application
        if r_val >= self.tau_r:
            # Gate 2: Structural Quarantine
            # Determine the anchor node from historical nodes [0 ... t-1]
            # We look for the maximum probability in a_nodes
            anchor_idx = int(a_probs.argmax())
            
            return {
                'status': 'BACKTRACK',
                'anchor_node': anchor_idx,
                'risk_score': r_val,
                'anchor_probs': a_probs.tolist()
            }
            
        elif r_val >= self.tau_v:
            # Gate 1: Semantic Retry
            return {
                'status': 'RETRY_STEP',
                'risk_score': r_val
            }
            
        else:
            # Pass
            return {
                'status': 'PASS',
                'risk_score': r_val
            }

if __name__ == "__main__":
    import os
    print("Initializing Controller for Smoke Test...")
    
    # We will try to load a known good model if it exists, otherwise randomly initialize
    model_path = 'best_parg_model.pt' if os.path.exists('best_parg_model.pt') else None
    if model_path:
        print(f"Loading trained weights from {model_path}...")
    else:
        print("No trained weights found. Using random initialization for demonstration.")
        
    controller = DualGateController(model_path=model_path, device='cpu')
    
    # Let's test it on a classic poisoned trajectory sequence
    mock_traj = [
        {'action': 'thought', 'content': 'Analyzing the query regarding capital cities.'},
        {'action': 'retrieve_memory', 'content': 'The capital of Australia is Sydney.'},
        {'action': 'thought', 'content': 'Comparing retrieved sources with internal knowledge.'},
        {'action': 'prioritize_recent_fact', 'content': 'The capital of Australia is Sydney.'}
    ]
    
    # 1. Test step by step
    print("\n--- Online Inference Simulation ---")
    current_prefix = []
    for step in mock_traj:
        current_prefix.append(step)
        print(f"\nEvaluating Step {len(current_prefix)}: {step['action']}")
        
        result = controller.evaluate_state(current_prefix)
        print(f"Risk Score (R_t): {result['risk_score']:.4f}")
        print(f"Action Taken: {result['status']}")
        
        if result['status'] == 'BACKTRACK':
            print(f"--> TRIGGER: Memory Firewall activated! Origin of corruption located at Node {result['anchor_node']}")
            print(f"--> True Anchor Probabilities: {[round(p, 4) for p in result['anchor_probs']]}")
            break
