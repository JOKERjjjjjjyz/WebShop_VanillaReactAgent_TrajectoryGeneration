import torch
import torch.nn as nn
try:
    from torch_geometric.nn import GATConv, global_mean_pool
except ImportError:
    # Fallback/Mock just in case torch_geometric is not installed in the environment yet
    # We will raise a warning and provide dummy classes, but expect this to be run where PyTorch Geometric is installed.
    import warnings
    warnings.warn("torch_geometric not installed. Using mock GAT blocks.")
    class GATConv(nn.Module):
        def __init__(self, in_channels, out_channels, heads=1, concat=True):
            super().__init__()
            self.linear = nn.Linear(in_channels, out_channels * heads)
        def forward(self, x, edge_index):
            return self.linear(x)
    def global_mean_pool(x, batch):
        if batch is None: return x.mean(dim=0, keepdim=True)
        # Proper fallback implementation
        batch_size = int(batch.max().item() + 1)
        out = torch.zeros(batch_size, x.size(-1), device=x.device)
        count = torch.zeros(batch_size, 1, device=x.device)
        out.scatter_add_(0, batch.unsqueeze(1).expand(-1, x.size(-1)), x)
        count.scatter_add_(0, batch.unsqueeze(1), torch.ones_like(batch.unsqueeze(1), dtype=torch.float, device=x.device))
        return out / count.clamp(min=1)

class SemanticStream(nn.Module):
    def __init__(self, embed_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        # Transformer Encoder for sequential trajectory history
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
    def forward(self, x_seq):
        """
        x_seq: (batch_size, seq_len, embed_dim)
        """
        out = self.transformer(x_seq)
        # Return the unpooled sequence for node-wise fusion
        return out

class TopologyStream(nn.Module):
    def __init__(self, node_dim=128, hidden_dim=64):
        super().__init__()
        # Graph Attention Network resolving step dependency bottlenecks
        self.gat1 = GATConv(node_dim, hidden_dim, heads=2, concat=True)
        self.gat2 = GATConv(hidden_dim * 2, hidden_dim, heads=1, concat=False)
        
    def forward(self, x, edge_index, batch=None):
        """
        x: (num_nodes, node_dim)
        edge_index: (2, num_edges)
        batch: (num_nodes,) graph assignment vector for batching
        """
        x = self.gat1(x, edge_index)
        x = nn.functional.elu(x)
        x = self.gat2(x, edge_index)
        # Return graph unpooled nodes for node-wise fusion
        return x

class PARGModel(nn.Module):
    """
    Propagation-Aware Risk Gating (PARG) Network v2.0.
    Dual-stream architecture handling:
    1) Semantic sequential logic errors
    2) Topological propagation structures (dependency graphs)
    Dual heads for macro-gating and precise backtracking.
    """
    def __init__(self, embed_dim=128, node_dim=128, hidden_dim=64):
        super().__init__()
        self.semantic = SemanticStream(embed_dim=embed_dim)
        self.topology = TopologyStream(node_dim=node_dim, hidden_dim=hidden_dim)
        
        # Context module mapping (Depth, Memory Conflict Ratio, Tool Density)
        self.context_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU()
        )
        
        # Node-wise Fusion Module
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Head A: Decision Risk (applied to tip node)
        self.risk_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Head B: Anchor Localization (applied to all historical nodes)
        self.anchor_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, seq_x, seq_lens, graph_x, edge_index, context_x, graph_batch):
        """
        Calculates propagation risk continuously.
        seq_x: (batch, seq_len, embed_dim) Padded sequence representations
        seq_lens: (batch,) True sequence lengths
        graph_x: (num_nodes, node_dim) Flattened graph node representations
        edge_index: (2, num_edges) Tool-memory dependencies
        context_x: (batch, 3) Global features
        graph_batch: (num_nodes,) Graph assignment for each node
        """
        h_sem_padded = self.semantic(seq_x) # (batch, max_len, embed_dim)
        
        # Extract valid nodes from h_sem_padded to match graph_x shape
        h_sem_list = [h_sem_padded[i, :seq_lens[i]] for i in range(len(seq_lens))]
        h_sem_flat = torch.cat(h_sem_list, dim=0) # (num_nodes, embed_dim)
        
        h_top_flat = self.topology(graph_x, edge_index) # (num_nodes, hidden_dim)
        
        h_ctx = self.context_mlp(context_x) # (batch, 16)
        h_ctx_flat = h_ctx[graph_batch] # (num_nodes, 16)
        
        # Node-wise Fusion
        fused = torch.cat([h_sem_flat, h_top_flat, h_ctx_flat], dim=-1)
        h_fused = self.fusion(fused) # (num_nodes, 64)
        
        # Tip Nodes for Risk Head
        tip_indices = torch.cumsum(seq_lens, dim=0) - 1
        h_tip = h_fused[tip_indices] # (batch, 64)
        
        r_prop = self.risk_head(h_tip).squeeze(-1) # (batch,)
        a_nodes = self.anchor_head(h_fused).squeeze(-1) # (num_nodes,)
        
        return r_prop, a_nodes

if __name__ == "__main__":
    # Smoke test dimensions matching the formalization
    model = PARGModel()
    mock_seq = torch.randn(2, 5, 128)          # Batch x SeqLen x Dim
    mock_lens = torch.tensor([5, 5], dtype=torch.long)
    mock_nodes = torch.randn(10, 128)          # Nodes x Dim (2 graphs of 5 nodes)
    mock_edges = torch.tensor([[0, 1, 2, 5, 6], 
                               [1, 2, 3, 6, 8]], dtype=torch.long)
    mock_batch = torch.tensor([0,0,0,0,0, 1,1,1,1,1])
    mock_ctx = torch.randn(2, 3)
    
    r_prop, a_nodes = model(mock_seq, mock_lens, mock_nodes, mock_edges, mock_ctx, graph_batch=mock_batch)
    print("Risk score shape:", r_prop.shape)
    print("Scores:\n", r_prop.detach())
    print("Anchor localized shape:", a_nodes.shape)
    print("Anchor Scores:\n", a_nodes.detach())
