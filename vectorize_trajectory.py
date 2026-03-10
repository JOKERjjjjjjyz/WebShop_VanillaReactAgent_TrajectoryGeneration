import json
from pathlib import Path
import torch

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    class DummySentenceTransformer:
        """Mock embedding model if sentence_transformers isn't installed yet."""
        def __init__(self, model_name):
            self.dim = 128 if "mini" in model_name.lower() else 384
        def encode(self, texts, convert_to_tensor=True):
            if isinstance(texts, str):
                texts = [texts]
            return torch.randn(len(texts), self.dim)
    SentenceTransformer = DummySentenceTransformer

class TrajectoryVectorizer:
    """
    Converts raw text from Agent trajectories into fixed-dimensional vectors 
    suitable for the PARG Neural Network (Semantic and Topology streams).
    """
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Loading embedding model '{model_name}' on {self.device}...")
        self.encoder = SentenceTransformer(model_name)
        # Move to device if it's the real SentenceTransformer
        if hasattr(self.encoder, 'to'):
            self.encoder = self.encoder.to(self.device)

    def vectorize_node(self, node_dict):
        """
        Converts a single step (node) into an embedding vector.
        Expects keys like 'action_type', 'query', 'observation', etc.
        """
        # Formulate a condensed text representation of the node
        action_type = node_dict.get('action', 'THOUGHT')
        content = node_dict.get('content', '')
        
        # Format: [ACTION] Content
        text_repr = f"[{action_type.upper()}] {content}"
        
        # Generate embedding
        with torch.no_grad():
            emb = self.encoder.encode([text_repr], convert_to_tensor=True)
            
        return emb.cpu().squeeze(0) # Returns 1D tensor

    def vectorize_trajectory(self, trajectory):
        """
        Converts an entire chronological trajectory into a sequence tensor.
        trajectory: List of node dictionaries.
        Returns: (seq_len, embed_dim) tensor
        """
        if not trajectory:
            return torch.empty(0)
            
        node_texts = [f"[{n.get('action', 'THOUGHT').upper()}] {n.get('content', '')}" for n in trajectory]
        
        with torch.no_grad():
            embs = self.encoder.encode(node_texts, convert_to_tensor=True)
            
        return embs.cpu()

if __name__ == "__main__":
    # Smoke test the vectorizer
    print("Initializing vectorizer...")
    vectorizer = TrajectoryVectorizer()
    
    sample_trajectory = [
        {"action": "thought", "content": "I need to find the capital of France."},
        {"action": "tool_call", "content": "Search(query='capital of France')"},
        {"action": "observation", "content": "Paris is the capital and most populous city of France."},
        {"action": "answer", "content": "The capital is Paris."}
    ]
    
    print("\nVectorizing trajectory of length 4...")
    seq_tensor = vectorizer.vectorize_trajectory(sample_trajectory)
    
    print(f"Resulting Tensor Shape: {seq_tensor.shape}")
    print(f"Showing first 5 dimensions of step 0: {seq_tensor[0][:5]}")
