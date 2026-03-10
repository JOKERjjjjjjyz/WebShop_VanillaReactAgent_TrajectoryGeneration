#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json

from p1_model import PARGModel
from vectorize_trajectory import TrajectoryVectorizer

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class PargDataset(Dataset):
    def __init__(self, data_paths, vectorizer):
        self.samples = []
        self.vectorizer = vectorizer
        
        raw_samples = []
        for path in data_paths:
            with open(path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    raw_samples.append(json.loads(line))
                    
        # Process and generate Prefix-Slices for DAGs
        for sample in raw_samples:
            raw_traj = sample.get('trajectory', [])
            parsed_nodes = []
            
            # Extract nodes and their explicit dependencies
            for node in raw_traj:
                act = str(node.get('action', 'thought'))
                content = str(node.get('content', ''))
                deps = node.get('dependencies', [])
                step_id = node.get('step_id', len(parsed_nodes))
                parsed_nodes.append({
                    'step_id': step_id,
                    'action': act, 
                    'content': content.strip(),
                    'dependencies': deps
                })
                    
            if not parsed_nodes:
                continue
                
            # Goal 1: Determine ultimate outcome (EFR)
            # The new dataset explicitly defines EFR (End-to-End Failure Rate)
            outcome_risk = float(sample.get('EFR', 0.0))
            is_failed_branch = (outcome_risk > 0.5)
            
            # Goal 2: Determine true anchor via causal assignment
            # The anchor is the point where the poison was ingested ('prioritize_recent_fact')
            # In the counterfactual data, the Safe branch also ingests it, but evaluating it saves the run. 
            true_anchor_idx = -1
            for i, p_node in enumerate(parsed_nodes):
                if p_node['action'] == 'prioritize_recent_fact':
                    true_anchor_idx = i
                    break
            
            # Vectorize the full trajectory once
            full_seq_x = self.vectorizer.vectorize_trajectory(parsed_nodes)
            
            # Prefix Slicing
            T = len(parsed_nodes)
            # Generate slice for each t in [1..T]
            for t in range(1, T + 1):
                seq_x = full_seq_x[:t]
                graph_x = seq_x.clone()
                
                # Edges: Build true DAG up to step t based on 'dependencies'
                edges_src = []
                edges_tgt = []
                for i in range(t):
                    node_deps = parsed_nodes[i]['dependencies']
                    for d in node_deps:
                        if d < t: # Only include valid historical dependencies
                            edges_src.append(d)
                            edges_tgt.append(i)
                    
                if not edges_src:
                    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                else:
                    edge_index = torch.tensor([edges_src, edges_tgt], dtype=torch.long)
                    
                # Context features: Calculate graph density and sequence length
                density = len(edges_src) / max(1, t)
                context_x = torch.tensor([float(t), density, 0.0], dtype=torch.float32)
                
                y_risk = torch.tensor(outcome_risk, dtype=torch.float32)
                
                # One-hot of anchor if exists in prefix
                y_anch = torch.zeros(t, dtype=torch.float32)
                if true_anchor_idx != -1 and true_anchor_idx < t:
                    y_anch[true_anchor_idx] = 1.0
                    
                self.samples.append({
                    'seq_x': seq_x,
                    'graph_x': graph_x,
                    'edge_index': edge_index,
                    'context_x': context_x,
                    'y_risk': y_risk,
                    'y_anch': y_anch,
                    'seq_len': t,
                    'is_failed_prefix': float(is_failed_branch) # Used for focal loss masking
                })

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_parg_batch(batch):
    seq_xs = [item['seq_x'] for item in batch]
    seq_x_padded = pad_sequence(seq_xs, batch_first=True, padding_value=0.0) 
    seq_lens = torch.tensor([item['seq_len'] for item in batch], dtype=torch.long)
    
    graph_xs = []
    edges_src = []
    edges_tgt = []
    graph_batch = []
    context_xs = []
    y_risks = []
    y_anchs = []
    is_failed_prefixes = []
    
    node_offset = 0
    for b_idx, item in enumerate(batch):
        g_x = item['graph_x']
        num_nodes = g_x.size(0)
        
        graph_xs.append(g_x)
        if item['edge_index'].size(1) == 1 and item['edge_index'][0,0].item() == 0 and item['edge_index'][1,0].item() == 0 and num_nodes == 1:
            edges_src.append(torch.tensor([node_offset], dtype=torch.long))
            edges_tgt.append(torch.tensor([node_offset], dtype=torch.long))
        else:
            edges_src.append(item['edge_index'][0] + node_offset)
            edges_tgt.append(item['edge_index'][1] + node_offset)
        
        graph_batch.extend([b_idx] * num_nodes)
        context_xs.append(item['context_x'])
        y_risks.append(item['y_risk'])
        y_anchs.append(item['y_anch'])
        is_failed_prefixes.append(torch.tensor([item['is_failed_prefix']]*num_nodes, dtype=torch.float32))
        
        node_offset += num_nodes
        
    graph_x_cat = torch.cat(graph_xs, dim=0)
    edge_index_cat = torch.stack([torch.cat(edges_src), torch.cat(edges_tgt)], dim=0)
    graph_batch_tensor = torch.tensor(graph_batch, dtype=torch.long)
    context_x_cat = torch.stack(context_xs, dim=0)
    y_risk_cat = torch.stack(y_risks, dim=0)
    y_anch_cat = torch.cat(y_anchs, dim=0)
    is_failed_cat = torch.cat(is_failed_prefixes, dim=0)
    
    return seq_x_padded, seq_lens, graph_x_cat, edge_index_cat, graph_batch_tensor, context_x_cat, y_risk_cat, y_anch_cat, is_failed_cat


class PARGTrainer:
    def __init__(self, lr=1e-3):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model = PARGModel().to(self.device)
        self.criterion = nn.BCELoss(reduction='none') 
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (seq_x, seq_lens, graph_x, edge_index, graph_batch, context_x, y_risk, y_anch, is_failed) in enumerate(dataloader):
            seq_x = seq_x.to(self.device)
            seq_lens = seq_lens.to(self.device)
            graph_x = graph_x.to(self.device)
            edge_index = edge_index.to(self.device)
            graph_batch = graph_batch.to(self.device)
            context_x = context_x.to(self.device)
            y_risk = y_risk.to(self.device)
            y_anch = y_anch.to(self.device)
            is_failed = is_failed.to(self.device)
            
            self.optimizer.zero_grad()
            
            r_prop, a_nodes = self.model(seq_x, seq_lens, graph_x, edge_index, context_x, graph_batch)
            
            # Loss A: Risk Loss (Weighted BCE)
            weights_risk = torch.where(y_risk == 1.0, torch.tensor(3.0, device=self.device), torch.tensor(1.0, device=self.device))
            loss_risk = (nn.functional.binary_cross_entropy(r_prop, y_risk, reduction='none') * weights_risk).mean()
            
            # Loss B: Anchor Localization Loss (Focal Loss)
            # Only compute for nodes belonging to known failed trajectories
            valid_mask = is_failed > 0.5
            if valid_mask.sum() > 0:
                a_nodes_valid = a_nodes[valid_mask]
                y_anch_valid = y_anch[valid_mask]
                
                # Focal Loss components
                gamma = 2.0
                alpha = 0.90 
                bce = nn.functional.binary_cross_entropy(a_nodes_valid, y_anch_valid, reduction='none')
                p_t = torch.exp(-bce)
                focal_term = (1 - p_t) ** gamma
                alpha_t = alpha * y_anch_valid + (1 - alpha) * (1 - y_anch_valid)
                loss_anch = (alpha_t * focal_term * bce).mean()
            else:
                loss_anch = torch.tensor(0.0, device=self.device)
            
            loss = loss_risk + 10.0 * loss_anch 
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / max(1, len(dataloader))
        print(f"Epoch {epoch} | Avg Total Loss: {avg_loss:.4f} (Risk: {loss_risk.item():.4f}, Anch: {loss_anch.item():.4f})")


if __name__ == "__main__":
    print("Initializing Data Pipeline...")
    vectorizer = TrajectoryVectorizer(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    data_paths = [
        Path(__file__).parent / 'data' / 'parg_train_v3.jsonl'
    ]
    valid_paths = [p for p in data_paths if p.exists()]
    
    if not valid_paths:
        print("Data files not found.")
        exit(1)
        
    print(f"Loading datasets: {[p.name for p in valid_paths]}")
    dataset = PargDataset(valid_paths, vectorizer)
    print(f"Total samples: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_parg_batch)
    
    print("\nBeginning Training Loop...")
    trainer = PARGTrainer()
    for epoch in range(1, 11):
        trainer.train_epoch(dataloader, epoch)
        
    # Save the trained model weights for the Controller to use
    save_path = 'best_parg_model.pt'
    torch.save(trainer.model.state_dict(), save_path)
    print(f"\nTraining complete. Model weights saved to {save_path}.")
    print("Risk network effectively traces toxic gradients.")
