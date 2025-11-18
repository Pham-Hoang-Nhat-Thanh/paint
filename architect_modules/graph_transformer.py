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
        # Convert layer positions to indices (scale by available positions) and
        # ensure indices are on the same device as the positional embeddings.
        max_pos = self.pe.shape[1] - 1
        pos_indices = (layer_positions * max_pos).long().clamp(0, max_pos).to(self.pe.device)

        # Get positional encodings
        pos_encoding = self.pe[0, pos_indices]  # [batch_size, num_nodes, hidden_dim]
        
        return node_features + pos_encoding

class EdgeAwareAttention(nn.Module):
    """Multi-head attention with edge feature integration
    
    Supports both dense masked attention and sparse edge-indexed attention via torch-scatter.
    Automatically selects the more efficient implementation based on graph density.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1, 
                 use_sparse_attention: bool = True):
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

        # Output projection
        self.output_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Note: LayerNorm is handled by the outer GraphTransformer per-layer
        # to avoid double residual+LayerNorm application. Do not keep an internal
        # layer_norm here.
        # Precompute attention scale to avoid repeated sqrt during forward
        self.scale = 1.0 / math.sqrt(self.head_dim)
        # Edge encoders (optional) -- will be configured lazily when edge feature dim is known
        # edge_bias_encoder: maps edge features -> per-head scalar bias
        # edge_v_encoder: maps edge features -> per-head vector to modulate V (num_heads * head_dim)
        self.edge_feat_dim = None
        self.edge_bias_encoder = None
        self.edge_v_encoder = None
        
    def create_adjacency_mask(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Create adjacency matrix mask from edge indices"""
        mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
        mask[edge_index[0], edge_index[1]] = True
        return mask
    
    def _sparse_attention_torch_scatter(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                        edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Industry-grade sparse edge-indexed attention using torch_scatter.

        Optimized for performance and memory efficiency with torch.compile.
        Uses scatter_softmax for O(E) complexity softmax computation.
        Advanced indexing for aggregation (memory efficient).

        Input shapes (after reshaping to multi-head):
        - Q, K, V: [batch_size, num_heads, num_nodes, head_dim]
        - edge_index: [2, num_edges]

        Returns: [batch_size, num_heads, num_nodes, head_dim]
        """
        batch_size, num_heads, _, head_dim = Q.shape
        device = Q.device
        num_edges = edge_index.shape[1]

        if num_edges == 0:
            return torch.zeros(batch_size, num_heads, num_nodes, head_dim, device=device)

        # Ensure index tensors live on the same device as the tensors used for indexing
        src_idx = edge_index[0].to(device)  # [num_edges]

        # Gather Q, K, V for each edge using advanced indexing
        Q_e = Q[:, :, src_idx, :]  # [batch_size, num_heads, num_edges, head_dim]
        K_e = K[:, :, edge_index[1], :]  # [batch_size, num_heads, num_edges, head_dim]
        V_e = V[:, :, edge_index[1], :]  # [batch_size, num_heads, num_edges, head_dim]

        # Compute attention scores: Q·K with scaling using einsum for optimization
        scores_e = torch.einsum('bhnd,bhnd->bhn', Q_e, K_e) * self.scale # [batch_size, num_heads, num_edges]

        # If edge encoders exist and edge features were pre-encoded into module attributes,
        # add per-edge per-head bias to scores and modulate V_e.
        if hasattr(self, '_edge_features') and self._edge_features is not None and self.edge_bias_encoder is not None:
            # _edge_features is expected shape [num_edges, feat_dim]
            ef = self._edge_features.to(device)
            # bias: [num_edges, num_heads] -> [1, num_heads, num_edges]
            bias = self.edge_bias_encoder(ef).transpose(0, 1).unsqueeze(0)
            scores_e = scores_e + bias
        if hasattr(self, '_edge_features') and self._edge_features is not None and self.edge_v_encoder is not None:
            ef = self._edge_features.to(device)
            # v_mod: [num_edges, num_heads * head_dim] -> [num_edges, num_heads, head_dim]
            vmod = self.edge_v_encoder(ef).view(-1, self.num_heads, head_dim)
            # vmod: [E, H, D] -> permute to [H, E, D] and unsqueeze batch dim -> [1, H, E, D]
            vmod = vmod.permute(1, 0, 2).unsqueeze(0).to(V_e.dtype)
            # Add vmod to V_e (broadcast over batch)
            V_e = V_e + vmod

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

        # Aggregate per source node using scatter_add to correctly handle duplicate indices
        aggregated_flat = torch.zeros(batch_size * num_heads, num_nodes, head_dim, device=device, dtype=V_flat.dtype)
        # Build index tensor for scatter_add: [B*H, E, D] -> index shape [B*H, E, D]
        index = src_idx_expanded.unsqueeze(-1).expand(batch_size * num_heads, num_edges, head_dim)
        aggregated_flat = aggregated_flat.scatter_add_(1, index, weighted_V_flat)

        # Reshape back to multi-head format
        attn_output = aggregated_flat.reshape(batch_size, num_heads, num_nodes, head_dim)

        return attn_output

    def _sparse_attention_torch_scatter_per_batch(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                                  edge_index_list, edge_features_list, num_nodes: int) -> torch.Tensor:
        """Handle a list of per-batch edge_index tensors by calling the single-graph
        sparse routine for each batch entry and stacking results. This preserves
        correct semantics when each graph has a different topology.
        """
        batch_size = Q.shape[0]
        out_list = []
        for b in range(batch_size):
            q_b = Q[b:b+1]  # [1, H, N, D]
            k_b = K[b:b+1]
            v_b = V[b:b+1]
            ei = edge_index_list[b]
            # Temporarily set per-module edge features for this single-graph call
            prev_ef = getattr(self, '_edge_features', None)
            self._edge_features = None if edge_features_list is None else edge_features_list[b].to(q_b.device)
            out_b = self._sparse_attention_torch_scatter(q_b, k_b, v_b, ei, num_nodes)
            # Restore
            self._edge_features = prev_ef
            out_list.append(out_b)
        return torch.cat(out_list, dim=0)
 
    def _sparse_attention_pure_pytorch(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                       edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Sparse edge-indexed attention using pure PyTorch (no torch_scatter dependency).
        
        Slower than torch_scatter but avoids external dependency.
        Groups edges by source node and computes softmax per group.
        """
        batch_size, num_heads, _, head_dim = Q.shape
        device = Q.device
        num_edges = edge_index.shape[1]
        
        if num_edges == 0:
            return torch.zeros(batch_size, num_heads, num_nodes, head_dim, device=device)
        
        # Ensure index tensors are on the same device as the embeddings
        src_idx = edge_index[0].to(device)  # [num_edges]
        
        # Gather Q and K for each edge
        Q_e = Q[:, :, src_idx, :]  # [batch_size, num_heads, num_edges, head_dim]
        K_e = K[:, :, edge_index[1], :]  # [batch_size, num_heads, num_edges, head_dim]
        V_e = V[:, :, edge_index[1], :]  # [batch_size, num_heads, num_edges, head_dim]
        
        # Compute per-edge scores
        scores_e = (Q_e * K_e).sum(-1) * self.scale  # [batch_size, num_heads, num_edges]
        # Add per-edge bias if available
        if hasattr(self, '_edge_features') and self._edge_features is not None and self.edge_bias_encoder is not None:
            ef = self._edge_features.to(device)
            bias = self.edge_bias_encoder(ef).transpose(0, 1).unsqueeze(0)  # [1, H, E]
            scores_e = scores_e + bias
        if hasattr(self, '_edge_features') and self._edge_features is not None and self.edge_v_encoder is not None:
            ef = self._edge_features.to(device)
            vmod = self.edge_v_encoder(ef).view(-1, self.num_heads, head_dim)
            vmod = vmod.permute(1, 0, 2).unsqueeze(0).to(V_e.dtype)
            V_e = V_e + vmod
        
        # Initialize output
        attn_output = torch.zeros(batch_size, num_heads, num_nodes, head_dim, device=device)
        
        # Group edges by source node and compute softmax per group
        unique_src = torch.unique(src_idx)

        # Loop over unique sources (kept as Python-level loop; consider vectorizing via scatter)
        for src in unique_src.tolist():
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
            
            # Store in output at source node position (src is int)
            attn_output[:, :, int(src), :] = aggregated
        
        return attn_output

    def _sparse_attention_pure_pytorch_per_batch(self, Q, K, V, edge_index_list, num_nodes: int):
        batch_size = Q.shape[0]
        out_list = []
        for b in range(batch_size):
            q_b = Q[b:b+1]
            k_b = K[b:b+1]
            v_b = V[b:b+1]
            ei = edge_index_list[b]
            prev_ef = getattr(self, '_edge_features', None)
            self._edge_features = None
            out_b = self._sparse_attention_pure_pytorch(q_b, k_b, v_b, ei, num_nodes)
            self._edge_features = prev_ef
            out_list.append(out_b)
        return torch.cat(out_list, dim=0)
    
    def forward(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor, edge_features=None) -> torch.Tensor:
        """
        node_embeddings: [batch_size, num_nodes, hidden_dim]
        edge_index: [2, num_edges]
        
        Automatically selects between sparse (torch_scatter) and dense masked attention.
        Sparse attention computes only over edges (O(E) complexity).
        Dense attention computes full QK^T and masks non-edges (O(N^2) complexity).
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
        # Support per-batch edge_index provided as a list/tuple of tensors
        is_per_batch = isinstance(edge_index, (list, tuple))
        # Attach edge features to module for use in lower-level routines
        if edge_features is not None:
            # edge_features may be a tensor [num_edges, feat_dim] or a list matching per-batch edge_index
            if is_per_batch and isinstance(edge_features, (list, tuple)):
                self._edge_features = None
                edge_features_list = edge_features
            else:
                self._edge_features = edge_features.to(node_embeddings.device)
                edge_features_list = None
            # Lazily configure encoders if needed
            if self.edge_feat_dim is None and self._edge_features is not None and self._edge_features.numel() > 0:
                self.edge_feat_dim = self._edge_features.shape[1]
                # Create encoders
                self.edge_bias_encoder = nn.Linear(self.edge_feat_dim, self.num_heads, bias=False).to(node_embeddings.device)
                self.edge_v_encoder = nn.Linear(self.edge_feat_dim, self.num_heads * self.head_dim, bias=False).to(node_embeddings.device)
        else:
            self._edge_features = None
            edge_features_list = None
        if self.use_sparse_attention and self.has_torch_scatter and (is_per_batch or (edge_index.shape[1] > 0)):
            if is_per_batch:
                attn_output = self._sparse_attention_torch_scatter_per_batch(Q, K, V, edge_index, edge_features_list, num_nodes)
            else:
                attn_output = self._sparse_attention_torch_scatter(Q, K, V, edge_index, num_nodes)
        elif self.use_sparse_attention and not self.has_torch_scatter and (is_per_batch or (edge_index.shape[1] > 0)):
            if is_per_batch:
                attn_output = self._sparse_attention_pure_pytorch_per_batch(Q, K, V, edge_index, num_nodes)
            else:
                attn_output = self._sparse_attention_pure_pytorch(Q, K, V, edge_index, num_nodes)
        else:
            # Use dense masked attention (original implementation) - dense mask can be per-batch
            attn_output = self._dense_attention_masked(Q, K, V, edge_index, num_nodes, edge_features_list)
        
        # Concatenate heads - [batch_size, num_nodes, hidden_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, num_nodes, hidden_dim
        )
        
        # Output projection
        output = self.output_linear(attn_output)
        output = self.dropout(output)
        # Return projected output (residual + layer-norm is applied by GraphTransformer)
        return output
    
    def _dense_attention_masked(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                edge_index: torch.Tensor, num_nodes: int, edge_features_list=None) -> torch.Tensor:
        """Dense masked attention (original implementation): computes full QK^T and masks non-edges.
        
        Input shapes (after reshaping to multi-head):
        - Q, K, V: [batch_size, num_heads, num_nodes, head_dim]
        - edge_index: [2, num_edges]
        
        Returns: [batch_size, num_heads, num_nodes, head_dim]
        """
        device = Q.device
        batch_size = Q.shape[0]
        
        # Compute attention scores: Q @ K^T
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # [batch_size, num_heads, num_nodes, num_nodes]
        
        # Use dtype-safe large negative value for masking (avoids fp16 overflow)
        large_neg = torch.finfo(attn_scores.dtype).min / 2

        # Support per-batch edge_index (list/tuple) or a single edge_index tensor
        if isinstance(edge_index, (list, tuple)):
            # Build per-batch mask: [batch_size, 1, num_nodes, num_nodes]
            mask = torch.zeros((batch_size, 1, num_nodes, num_nodes), dtype=torch.bool, device=device)
            for b, ei in enumerate(edge_index):
                if ei is None or ei.shape[1] == 0:
                    continue
                src = ei[0].to(device)
                tgt = ei[1].to(device)
                mask[b, 0, src, tgt] = True
            attn_scores = attn_scores.masked_fill(~mask, large_neg)
            # If per-batch edge features provided, apply biases and V modulation per batch
            if edge_features_list is not None and self.edge_bias_encoder is not None:
                # edge_features_list is list of [E_b, feat_dim]
                for b, ef in enumerate(edge_features_list):
                    if ef is None or ef.shape[0] == 0:
                        continue
                    ef = ef.to(device)
                    ei = edge_index[b]
                    src = ei[0].to(device)
                    tgt = ei[1].to(device)
                    bias = self.edge_bias_encoder(ef).transpose(0,1)  # [H, E]
                    for e_i, (s_i, t_i) in enumerate(zip(src.tolist(), tgt.tolist())):
                        attn_scores[b, :, s_i, t_i] = attn_scores[b, :, s_i, t_i] + bias[:, e_i]
                    # V modulation
                    if self.edge_v_encoder is not None:
                        vmod = self.edge_v_encoder(ef).view(-1, self.num_heads, self.head_dim)
                        # Add vmod to target positions in V
                        for e_i, t_i in enumerate(tgt.tolist()):
                            V[b, :, t_i, :] = V[b, :, t_i, :] + vmod[e_i]
        else:
            if edge_index.shape[1] > 0:
                # Build mask: [num_nodes, num_nodes]
                mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=device)
                mask[edge_index[0].to(device), edge_index[1].to(device)] = True
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, num_nodes, num_nodes]
                # Apply mask: set non-edges to a large negative constant
                attn_scores = attn_scores.masked_fill(~mask, large_neg)
                # Apply edge feature based bias/modulation for single-graph case
                if hasattr(self, '_edge_features') and self._edge_features is not None and self.edge_bias_encoder is not None:
                    ef = self._edge_features.to(device)
                    src_idx = edge_index[0].to(device)
                    tgt_idx = edge_index[1].to(device)
                    bias = self.edge_bias_encoder(ef).transpose(0,1)  # [H, E]
                    # Add bias to attn_scores at positions (src,tgt)
                    for e_i, (s_i, t_i) in enumerate(zip(src_idx.tolist(), tgt_idx.tolist())):
                        attn_scores[:, :, s_i, t_i] = attn_scores[:, :, s_i, t_i] + bias[:, e_i]
                    # V modulation
                    if self.edge_v_encoder is not None:
                        vmod = self.edge_v_encoder(ef).view(-1, self.num_heads, self.head_dim)
                        for e_i, t_i in enumerate(tgt_idx.tolist()):
                            V[:, :, t_i, :] = V[:, :, t_i, :] + vmod[e_i].unsqueeze(0)
            else:
                # No edges: set to large negative
                attn_scores = torch.full_like(attn_scores, large_neg)
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [batch_size, num_heads, num_nodes, head_dim]
        
        return attn_output

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
        # Reduce pooled edge embedding (preserve vector information)
        self.structure_reduce = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        
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
            # Vectorized edge embedding creation: avoid Python loop over edges
            # edge_index: [2, num_edges]
            src_idx = edge_index[0].to(node_embeddings.device)
            tgt_idx = edge_index[1].to(node_embeddings.device)

            # index_select is efficient on tensors and avoids Python-level loops
            src_embeds = node_embeddings.index_select(1, src_idx)  # [batch_size, num_edges, hidden_dim]
            tgt_embeds = node_embeddings.index_select(1, tgt_idx)  # [batch_size, num_edges, hidden_dim]

            edge_embeddings = torch.cat([src_embeds, tgt_embeds], dim=-1)  # [batch_size, num_edges, hidden_dim * 2]

            # Compute attention weights for edges and pool
            edge_attention_weights = self.structure_attention(edge_embeddings)  # [batch_size, num_edges, 1]
            edge_attention_weights = F.softmax(edge_attention_weights, dim=1)

            # Pool full edge embeddings (preserve vector info), then reduce to hidden_dim
            structure_pooled_vec = torch.sum(edge_attention_weights * edge_embeddings, dim=1)  # [batch_size, hidden_dim*2]
            structure_pooled = self.structure_reduce(structure_pooled_vec)  # [batch_size, hidden_dim]
        else:
            # No edges, use zero padding
            structure_pooled = torch.zeros(batch_size, hidden_dim, device=node_embeddings.device)
        
        # 3. Combine both representations
        global_embedding = torch.cat([node_pooled, structure_pooled], dim=-1)
        
        return global_embedding

class GraphTransformer(nn.Module):
    """Complete Graph Transformer for neural architecture encoding"""
    
    def __init__(self, node_feature_dim: int, hidden_dim: int = 128, num_heads: int = 8, 
                 num_layers: int = 3, dropout: float = 0.1, use_sparse_attention: bool = True):
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
        
        # Layer norms: separate norms for attention output and FFN output per layer
        self.attn_layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.ffn_layer_norms = nn.ModuleList([
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
        Forward pass - handles both single and batched graphs.
        
        For single graphs (from MCTS/eval): Works as before
        For batched graphs (from training): Processes concatenated graph as single batch
        
        graph_data: {
            'node_features': [1, num_nodes, node_feature_dim],
            'edge_index': [2, num_edges], 
            'edge_weights': [num_edges],
            'layer_positions': [1, num_nodes]
            'batch': [total_nodes] - optional, for batching info (ignored here)
            'num_graphs': int - optional, for batching info (ignored here)
        }
        
        Returns: 
            global_embedding: [1, hidden_dim * 2]
            node_embeddings: [1, num_nodes, hidden_dim]
        """
        node_features = graph_data['node_features']
        edge_index = graph_data['edge_index']
        edge_features = graph_data.get('edge_features', None)
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
            attn_out = self.layers[i](x, edge_index, edge_features=edge_features)
            x = self.attn_layer_norms[i](residual + attn_out)

            # Feed-forward
            residual = x
            ffn_out = self.ffns[i](x)
            x = self.ffn_layer_norms[i](residual + ffn_out)
        
        node_embeddings = x  # Final node embeddings
        
        # Global pooling
        global_embedding = self.pooling(node_embeddings, edge_index)
        
        return global_embedding, node_embeddings