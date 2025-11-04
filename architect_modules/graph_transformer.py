import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple

# Enable TF32 for faster matrix multiplications on Ampere+ GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    torch.backends.cudnn.conv.fp32_precision = 'tf32'

class PositionalEncoding(nn.Module):
    """Positional encoding for neuron layer positions"""
    
    def __init__(self, hidden_dim, max_positions=1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Create positional encoding matrix
        pe = torch.zeros(max_positions, hidden_dim)
        position = torch.arange(0, max_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                            (-math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_positions, hidden_dim]
        
        self.register_buffer('pe', pe)
    
    def forward(self, node_features, layer_positions):
        """
        node_features: [batch_size, num_nodes, hidden_dim]
        layer_positions: [batch_size, num_nodes] normalized positions (0-1)
        """
        # Convert layer positions to indices (0-999)
        pos_indices = (layer_positions * 999).long().clamp(0, 999)
        
        # Get positional encodings
        pos_encoding = self.pe[0, pos_indices]  # [batch_size, num_nodes, hidden_dim]
        
        return node_features + pos_encoding

class EdgeAwareAttention(nn.Module):
    """Multi-head attention with edge feature integration"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        # Node transformations - use fused operations where possible
        self.q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)  # Remove bias for speed
        self.k_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Edge feature transformation
        self.edge_encoder = nn.Linear(1, num_heads, bias=False)

        # Output projection
        self.output_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def create_adjacency_mask(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Create adjacency matrix mask from edge indices"""
        mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
        mask[edge_index[0], edge_index[1]] = True
        return mask
    
    def forward(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor, 
                edge_weights: torch.Tensor) -> torch.Tensor:
        """
        node_embeddings: [batch_size, num_nodes, hidden_dim]
        edge_index: [2, num_edges]
        edge_weights: [num_edges]
        
        Uses masked attention: computes full O(n²) attention but zeros out non-edges.
        More efficient than full dense for sparse graphs while maintaining exact equivalence.
        """
        batch_size, num_nodes, hidden_dim = node_embeddings.shape
        device = node_embeddings.device
        
        # Store residual
        residual = node_embeddings
        
        # Linear projections
        Q = self.q_linear(node_embeddings)  # [batch_size, num_nodes, hidden_dim]
        K = self.k_linear(node_embeddings)
        V = self.v_linear(node_embeddings)
        
        # Reshape for multi-head attention - [batch_size, num_heads, num_nodes, head_dim]
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # [batch_size, num_heads, num_nodes, num_nodes]
        
        # Create adjacency mask from edge indices
        if edge_index.shape[1] > 0:
            # Build mask: [num_nodes, num_nodes]
            mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=device)
            mask[edge_index[0], edge_index[1]] = True
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, num_nodes, num_nodes]
            
            # Apply mask: set non-edges to -inf
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        else:
            # No edges at all - mask everything
            attn_scores = torch.full_like(attn_scores, float('-inf'))
        
        # Softmax with NaN handling (nodes with no incoming edges get NaN, convert to 0)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)
        
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [batch_size, num_heads, num_nodes, head_dim]
        
        # Concatenate heads - [batch_size, num_nodes, hidden_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, num_nodes, hidden_dim
        )
        
        # Output projection
        output = self.output_linear(attn_output)
        output = self.dropout(output)
        
        # Add residual and layer norm
        output = self.layer_norm(output + residual)
        
        return output

class HierarchicalPooling(nn.Module):
    """Hierarchical attention pooling for graph-level representation"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Node-level attention
        self.node_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Structure-level attention (for edge patterns)
        self.structure_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(), 
            nn.Linear(hidden_dim, 1)
        )
        
        # Global context vector
        self.global_context = nn.Parameter(torch.randn(hidden_dim))
        
    def forward(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        node_embeddings: [batch_size, num_nodes, hidden_dim]
        edge_index: [2, num_edges]
        Returns: [batch_size, hidden_dim * 2] (concatenated node and structure embeddings)
        """
        batch_size, num_nodes, hidden_dim = node_embeddings.shape
        
        # 1. Node-level pooling
        node_attention_weights = self.node_attention(node_embeddings)  # [batch_size, num_nodes, 1]
        node_attention_weights = F.softmax(node_attention_weights, dim=1)
        node_pooled = torch.sum(node_attention_weights * node_embeddings, dim=1)  # [batch_size, hidden_dim]
        
        # 2. Structure-aware pooling
        if edge_index.shape[1] > 0:
            edge_embeddings = []
            for i in range(edge_index.shape[1]):
                src, tgt = edge_index[0, i], edge_index[1, i]
                # For each edge, concatenate source and target embeddings
                edge_embed = torch.cat([
                    node_embeddings[:, src, :], 
                    node_embeddings[:, tgt, :]
                ], dim=-1)  # [batch_size, hidden_dim * 2]
                edge_embeddings.append(edge_embed)
            
            edge_embeddings = torch.stack(edge_embeddings, dim=1)  # [batch_size, num_edges, hidden_dim * 2]
            
            # Compute attention weights for edges
            edge_attention_weights = self.structure_attention(edge_embeddings)  # [batch_size, num_edges, 1]
            edge_attention_weights = F.softmax(edge_attention_weights, dim=1)
            
            # Pool edge embeddings
            structure_pooled = torch.sum(
                edge_attention_weights * edge_embeddings.mean(dim=-1, keepdim=True), 
                dim=1
            )  # [batch_size, 1]
            
            # Expand to match hidden_dim
            structure_pooled = structure_pooled.expand(-1, hidden_dim)
        else:
            # No edges, use zero padding
            structure_pooled = torch.zeros(batch_size, hidden_dim, device=node_embeddings.device)
        
        # 3. Combine both representations
        global_embedding = torch.cat([node_pooled, structure_pooled], dim=-1)
        
        return global_embedding

class GraphTransformer(nn.Module):
    """Complete Graph Transformer for neural architecture encoding"""
    
    def __init__(self, node_feature_dim: int, hidden_dim: int = 128, num_heads: int = 8, 
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection - optimized
        self.input_projection = nn.Linear(node_feature_dim, hidden_dim, bias=False)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            EdgeAwareAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Feed-forward networks - optimized for speed
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4, bias=False),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim, bias=False),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Pooling
        self.pooling = HierarchicalPooling(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, graph_data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        graph_data: {
            'node_features': [batch_size, num_nodes, node_feature_dim],
            'edge_index': [2, num_edges], 
            'edge_weights': [num_edges],
            'layer_positions': [batch_size, num_nodes]  # normalized 0-1
        }
        Returns: 
            global_embedding: [batch_size, hidden_dim * 2]
            node_embeddings: [batch_size, num_nodes, hidden_dim]
        """
        node_features = graph_data['node_features']
        edge_index = graph_data['edge_index']
        edge_weights = graph_data['edge_weights']
        layer_positions = graph_data['layer_positions']

        # Input projection
        x = self.input_projection(node_features)  # [batch_size, num_nodes, hidden_dim]
        x = self.dropout(x)
        
        # Add positional encoding
        x = self.positional_encoding(x, layer_positions)
        
        # Transformer layers
        for i in range(self.num_layers):
            # Self-attention
            residual = x
            x = self.layers[i](x, edge_index, edge_weights)
            x = self.layer_norms[i](x + residual)
            
            # Feed-forward
            residual = x
            x = self.ffns[i](x)
            x = self.layer_norms[i](x + residual)
        
        node_embeddings = x  # Final node embeddings
        
        # Global pooling
        global_embedding = self.pooling(node_embeddings, edge_index)
        
        return global_embedding, node_embeddings