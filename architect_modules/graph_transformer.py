import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple

# Try to import torch_scatter for efficient sparse attention
try:
    from torch_scatter import scatter_softmax
    HAS_TORCH_SCATTER = True
except ImportError:
    HAS_TORCH_SCATTER = False


# Enable TF32 for faster matrix multiplications on Ampere+ GPUs
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except AttributeError:
        # Older PyTorch versions
        pass

class PositionalEncoding(nn.Module):
    """Positional encoding for neuron layer positions.

    This module adds positional information to the node features based on their
    normalized layer positions within the neural architecture. It uses a
    sinusoidal encoding scheme.

    Attributes:
        hidden_dim (int): The dimensionality of the hidden features.
        pe (torch.Tensor): The pre-calculated positional encoding matrix.
    """
    
    def __init__(self, hidden_dim, max_positions=1000):
        """Initializes the PositionalEncoding module.

        Args:
            hidden_dim (int): The dimensionality of the hidden features.
            max_positions (int): The maximum number of positions to encode.
        """
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
        """Adds positional encoding to node features.

        Args:
            node_features (torch.Tensor): The input node features of shape
                [batch_size, num_nodes, hidden_dim].
            layer_positions (torch.Tensor): The normalized layer positions of
                nodes (0-1) of shape [batch_size, num_nodes].

        Returns:
            torch.Tensor: The node features with added positional encoding.
        """
        # Convert layer positions to indices (0-999)
        pos_indices = (layer_positions * 999).long().clamp(0, 999)
        
        # Get positional encodings
        pos_encoding = self.pe[0, pos_indices]  # [batch_size, num_nodes, hidden_dim]
        
        return node_features + pos_encoding

class EdgeAwareAttention(nn.Module):
    """Multi-head attention with edge feature integration.
    
    Supports both dense masked attention and sparse edge-indexed attention via
    torch-scatter. Automatically selects the more efficient implementation
    based on graph density.

    Attributes:
        hidden_dim (int): The dimensionality of the hidden features.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimensionality of each attention head.
        use_sparse_attention (bool): Whether to use sparse attention.
        has_torch_scatter (bool): Whether torch_scatter is available.
        q_linear (nn.Linear): Linear layer for queries.
        k_linear (nn.Linear): Linear layer for keys.
        v_linear (nn.Linear): Linear layer for values.
        edge_encoder (nn.Linear): Linear layer for edge features.
        output_linear (nn.Linear): Linear layer for the output.
        dropout (nn.Dropout): Dropout layer.
        layer_norm (nn.LayerNorm): Layer normalization.
        scale (float): The attention scaling factor.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1, 
                 use_sparse_attention: bool = True):
        """Initializes the EdgeAwareAttention module.

        Args:
            hidden_dim (int): The dimensionality of the hidden features.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
            use_sparse_attention (bool): Whether to use sparse attention.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        # Keep the user intent separate from which optional libraries are available.
        # This lets us pick the best available sparse implementation at runtime.
        self.use_sparse_attention = use_sparse_attention
        self.has_torch_scatter = HAS_TORCH_SCATTER

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
        # Precompute attention scale to avoid repeated sqrt during forward
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def create_adjacency_mask(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Create adjacency matrix mask from edge indices.

        Args:
            edge_index (torch.Tensor): The edge indices of shape [2, num_edges].
            num_nodes (int): The number of nodes in the graph.

        Returns:
            torch.Tensor: The adjacency matrix mask.
        """
        mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
        mask[edge_index[0], edge_index[1]] = True
        return mask
    
    def _sparse_attention_torch_scatter(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                        edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Industry-grade sparse edge-indexed attention using torch_scatter.

        Optimized for performance and memory efficiency with torch.compile.
        Uses scatter_softmax for O(E) complexity softmax computation.
        Advanced indexing for aggregation (memory efficient).

        Args:
            Q (torch.Tensor): Queries of shape [batch_size, num_heads, num_nodes, head_dim].
            K (torch.Tensor): Keys of shape [batch_size, num_heads, num_nodes, head_dim].
            V (torch.Tensor): Values of shape [batch_size, num_heads, num_nodes, head_dim].
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
            num_nodes (int): The number of nodes in the graph.

        Returns:
            torch.Tensor: The output of the attention mechanism of shape
                [batch_size, num_heads, num_nodes, head_dim].
        """
        batch_size, num_heads, _, head_dim = Q.shape
        device = Q.device
        num_edges = edge_index.shape[1]

        if num_edges == 0:
            return torch.zeros(batch_size, num_heads, num_nodes, head_dim, device=device)

        src_idx = edge_index[0]  # [num_edges]

        # Gather Q, K, V for each edge using advanced indexing
        Q_e = Q[:, :, src_idx, :]  # [batch_size, num_heads, num_edges, head_dim]
        K_e = K[:, :, edge_index[1], :]  # [batch_size, num_heads, num_edges, head_dim]
        V_e = V[:, :, edge_index[1], :]  # [batch_size, num_heads, num_edges, head_dim]

        # Compute attention scores: Q·K with scaling using einsum for optimization
        scores_e = torch.einsum('bhnd,bhnd->bhn', Q_e, K_e) * self.scale # [batch_size, num_heads, num_edges]

        # Flatten batch and head dimensions for scatter operations
        scores_flat = scores_e.reshape(batch_size * num_heads, num_edges)  # [B*H, E]
        V_flat = V_e.reshape(batch_size * num_heads, num_edges, head_dim)  # [B*H, E, D]

        # Expand src_idx for each batch-head combination
        src_idx_expanded = src_idx.unsqueeze(0).expand(batch_size * num_heads, -1)  # [B*H, E]

        # Compute per-source softmax using scatter_softmax (O(E) complexity)
        attn_weights_flat = scatter_softmax(scores_flat, src_idx_expanded, dim=1, dim_size=num_nodes)

        # Apply dropout to attention weights
        attn_weights_flat = self.dropout(attn_weights_flat)

        # Weight the values
        weighted_V_flat = attn_weights_flat.unsqueeze(-1) * V_flat # [B*H, E, D]

        # Aggregate per source node using advanced indexing (memory efficient)
        aggregated_flat = torch.zeros(batch_size * num_heads, num_nodes, head_dim, device=device)
        aggregated_flat[:, src_idx, :] += weighted_V_flat

        # Reshape back to multi-head format
        attn_output = aggregated_flat.reshape(batch_size, num_heads, num_nodes, head_dim)

        return attn_output
 
    def _sparse_attention_pure_pytorch(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                       edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Sparse edge-indexed attention using pure PyTorch.
        
        Slower than torch_scatter but avoids external dependency.
        Groups edges by source node and computes softmax per group.

        Args:
            Q (torch.Tensor): Queries of shape [batch_size, num_heads, num_nodes, head_dim].
            K (torch.Tensor): Keys of shape [batch_size, num_heads, num_nodes, head_dim].
            V (torch.Tensor): Values of shape [batch_size, num_heads, num_nodes, head_dim].
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
            num_nodes (int): The number of nodes in the graph.

        Returns:
            torch.Tensor: The output of the attention mechanism of shape
                [batch_size, num_heads, num_nodes, head_dim].
        """
        batch_size, num_heads, _, head_dim = Q.shape
        device = Q.device
        num_edges = edge_index.shape[1]
        
        if num_edges == 0:
            return torch.zeros(batch_size, num_heads, num_nodes, head_dim, device=device)
        
        src_idx = edge_index[0]  # [num_edges]
        
        # Gather Q and K for each edge
        Q_e = Q[:, :, src_idx, :]  # [batch_size, num_heads, num_edges, head_dim]
        K_e = K[:, :, edge_index[1], :]  # [batch_size, num_heads, num_edges, head_dim]
        V_e = V[:, :, edge_index[1], :]  # [batch_size, num_heads, num_edges, head_dim]
        
        # Compute per-edge scores
        scores_e = (Q_e * K_e).sum(-1) * self.scale  # [batch_size, num_heads, num_edges]
        
        # Initialize output
        attn_output = torch.zeros(batch_size, num_heads, num_nodes, head_dim, device=device)
        
        # Group edges by source node and compute softmax per group
        unique_src = torch.unique(src_idx)
        
        for src in unique_src:
            # Find all edges with this source node
            mask = (src_idx == src)
            
            # Extract scores and values for this group
            scores_group = scores_e[:, :, mask]  # [batch_size, num_heads, group_size]
            V_group = V_e[:, :, mask, :]  # [batch_size, num_heads, group_size, head_dim]
            
            # Compute softmax per group
            attn_weights_group = F.softmax(scores_group, dim=-1)
            attn_weights_group = self.dropout(attn_weights_group)
            
            # Aggregate V: sum over group dimension
            aggregated = torch.matmul(attn_weights_group.unsqueeze(-2), V_group)  
            # [batch_size, num_heads, 1, head_dim]
            aggregated = aggregated.squeeze(-2)  # [batch_size, num_heads, head_dim]
            
            # Store in output at source node position
            attn_output[:, :, src, :] = aggregated
        
        return attn_output
    
    def forward(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for EdgeAwareAttention.
        
        Automatically selects between sparse (torch_scatter) and dense masked attention.
        Sparse attention computes only over edges (O(E) complexity).
        Dense attention computes full QK^T and masks non-edges (O(N^2) complexity).

        Args:
            node_embeddings (torch.Tensor): Node embeddings of shape
                [batch_size, num_nodes, hidden_dim].
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges].

        Returns:
            torch.Tensor: The output of the attention mechanism.
        """
        batch_size, num_nodes, hidden_dim = node_embeddings.shape
        
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
        
        # Choose attention implementation based on graph density and available libraries.
        # Priority: torch_scatter (fast O(E) via scatter_softmax) -> Triton (custom kernel)
        # -> pure PyTorch sparse fallback -> dense masked attention.
        if self.use_sparse_attention and self.has_torch_scatter and edge_index.shape[1] > 0:
            # Use sparse edge-indexed attention via torch_scatter
            attn_output = self._sparse_attention_torch_scatter(Q, K, V, edge_index, num_nodes)
        elif self.use_sparse_attention and not self.has_torch_scatter and edge_index.shape[1] > 0:
            # Fallback to pure PyTorch sparse attention (slower but no dependency)
            attn_output = self._sparse_attention_pure_pytorch(Q, K, V, edge_index, num_nodes)
        else:
            # Use dense masked attention (original implementation)
            attn_output = self._dense_attention_masked(Q, K, V, edge_index, num_nodes)
        
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
    
    def _dense_attention_masked(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Dense masked attention: computes full QK^T and masks non-edges.
        
        Args:
            Q (torch.Tensor): Queries of shape [batch_size, num_heads, num_nodes, head_dim].
            K (torch.Tensor): Keys of shape [batch_size, num_heads, num_nodes, head_dim].
            V (torch.Tensor): Values of shape [batch_size, num_heads, num_nodes, head_dim].
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
            num_nodes (int): The number of nodes in the graph.
        
        Returns:
            torch.Tensor: The output of the attention mechanism of shape
                [batch_size, num_heads, num_nodes, head_dim].
        """
        device = Q.device
        
        # Compute attention scores: Q @ K^T
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # [batch_size, num_heads, num_nodes, num_nodes]
        
        # Create adjacency mask from edge indices
        if edge_index.shape[1] > 0:
            # Build mask: [num_nodes, num_nodes]
            mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=device)
            mask[edge_index[0], edge_index[1]] = True
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, num_nodes, num_nodes]
            
            # Apply mask: set non-edges to a large negative constant
            attn_scores = attn_scores.masked_fill(~mask, -1e9)
        else:
            # No edges: set to large negative
            attn_scores = torch.full_like(attn_scores, -1e9)
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [batch_size, num_heads, num_nodes, head_dim]
        
        return attn_output

class HierarchicalPooling(nn.Module):
    """Hierarchical attention pooling for graph-level representation.

    This module computes a graph-level embedding by pooling node and edge
    information using attention mechanisms.

    Attributes:
        hidden_dim (int): The dimensionality of the hidden features.
        node_attention (nn.Sequential): Attention mechanism for nodes.
        structure_attention (nn.Sequential): Attention mechanism for edges.
        global_context (nn.Parameter): A learnable global context vector.
    """
    
    def __init__(self, hidden_dim: int):
        """Initializes the HierarchicalPooling module.

        Args:
            hidden_dim (int): The dimensionality of the hidden features.
        """
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
        """Forward pass for HierarchicalPooling.

        Args:
            node_embeddings (torch.Tensor): Node embeddings of shape
                [batch_size, num_nodes, hidden_dim].
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges].

        Returns:
            torch.Tensor: The global graph embedding of shape
                [batch_size, hidden_dim * 2].
        """
        batch_size, num_nodes, hidden_dim = node_embeddings.shape
        
        # 1. Node-level pooling
        node_attention_weights = self.node_attention(node_embeddings)  # [batch_size, num_nodes, 1]
        node_attention_weights = F.softmax(node_attention_weights, dim=1)
        node_pooled = torch.sum(node_attention_weights * node_embeddings, dim=1)  # [batch_size, hidden_dim]
        
        # 2. Structure-aware pooling
        if edge_index.shape[1] > 0:
            # Vectorized edge embedding creation: avoid Python loop over edges
            # edge_index: [2, num_edges]
            src_idx = edge_index[0]
            tgt_idx = edge_index[1]

            # index_select is efficient on tensors and avoids Python-level loops
            src_embeds = node_embeddings.index_select(1, src_idx)  # [batch_size, num_edges, hidden_dim]
            tgt_embeds = node_embeddings.index_select(1, tgt_idx)  # [batch_size, num_edges, hidden_dim]

            edge_embeddings = torch.cat([src_embeds, tgt_embeds], dim=-1)  # [batch_size, num_edges, hidden_dim * 2]
            
            # Compute attention weights for edges and pool
            edge_attention_weights = self.structure_attention(edge_embeddings)  # [batch_size, num_edges, 1]
            edge_attention_weights = F.softmax(edge_attention_weights, dim=1)

            # The original implementation reduced the edge embeddings to a single value per-edge
            # and pooled those scalars using edge attention. Preserve that behaviour but
            # compute it vectorized and memory-efficiently.
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
    """Complete Graph Transformer for neural architecture encoding.

    This module combines edge-aware attention layers, feed-forward networks,
    and hierarchical pooling to create a comprehensive graph transformer model.

    Attributes:
        hidden_dim (int): The dimensionality of the hidden features.
        num_layers (int): The number of transformer layers.
        use_sparse_attention (bool): Whether to use sparse attention.
        input_projection (nn.Linear): Linear layer for input projection.
        positional_encoding (PositionalEncoding): Positional encoding module.
        layers (nn.ModuleList): List of EdgeAwareAttention layers.
        layer_norms (nn.ModuleList): List of LayerNorm layers.
        ffns (nn.ModuleList): List of feed-forward networks.
        pooling (HierarchicalPooling): Hierarchical pooling module.
        dropout (nn.Dropout): Dropout layer.
    """
    
    def __init__(self, node_feature_dim: int, hidden_dim: int = 128, num_heads: int = 8, 
                 num_layers: int = 3, dropout: float = 0.1, use_sparse_attention: bool = True):
        """Initializes the GraphTransformer module.

        Args:
            node_feature_dim (int): The dimensionality of the node features.
            hidden_dim (int): The dimensionality of the hidden features.
            num_heads (int): The number of attention heads.
            num_layers (int): The number of transformer layers.
            dropout (float): The dropout rate.
            use_sparse_attention (bool): Whether to use sparse attention.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_sparse_attention = use_sparse_attention
        
        # Input projection - optimized
        self.input_projection = nn.Linear(node_feature_dim, hidden_dim, bias=False)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            EdgeAwareAttention(hidden_dim, num_heads, dropout, use_sparse_attention=use_sparse_attention)
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
        """Forward pass for the GraphTransformer.
        
        Handles both single and batched graphs.

        Args:
            graph_data (Dict): A dictionary containing graph data with keys:
                'node_features', 'edge_index', 'layer_positions'.

        Returns: 
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the
                global graph embedding and the final node embeddings.
        """
        node_features = graph_data['node_features']
        edge_index = graph_data['edge_index']
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
            x = self.layers[i](x, edge_index)
            x = self.layer_norms[i](x + residual)
            
            # Feed-forward
            residual = x
            x = self.ffns[i](x)
            x = self.layer_norms[i](x + residual)
        
        node_embeddings = x  # Final node embeddings
        
        # Global pooling
        global_embedding = self.pooling(node_embeddings, edge_index)
        
        return global_embedding, node_embeddings
