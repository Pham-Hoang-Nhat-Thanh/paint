import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from .graph_transformer import GraphTransformer
import traceback
from typing import Dict
from blueprint_modules.network import NeuralArchitecture, NeuronType, ActivationType
from blueprint_modules.action import ActionType, Action, ActionSpace

# Enable TF32 for faster matrix multiplications on Ampere+ GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    torch.backends.cudnn.conv.fp32_precision = 'tf32'

class UnifiedPolicyValueNetwork(nn.Module):
    """Unified network that outputs both policy and value predictions"""
    
    def __init__(self, node_feature_dim: int, hidden_dim: int = 128,
                 max_neurons: int = 1000, num_actions: int = 5,
                 num_activations: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_neurons = max_neurons
        self.num_actions = num_actions
        self.num_activations = num_activations
        
        # Graph transformer backbone
        self.graph_transformer = GraphTransformer(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim
        )
        
        # Shared backbone MLP - optimized for speed (removed biases for faster computation)
        self.shared_backbone = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),  # global_embedding is hidden_dim * 2
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Policy heads (factorized action space) - no biases for faster computation
        self.action_type_head = nn.Linear(hidden_dim // 2, num_actions, bias=False)
        self.source_neuron_head = nn.Linear(hidden_dim // 2, max_neurons, bias=False)
        self.target_neuron_head = nn.Linear(hidden_dim // 2, max_neurons, bias=False)
        self.activation_head = nn.Linear(hidden_dim // 2, num_activations, bias=False)
        
        # Value head - optimized
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1, bias=False),
            nn.Tanh()  # Output between -1 and 1
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, graph_data: Dict, sub_batch_size: int = 4) -> Dict[str, torch.Tensor]:
        """
        Supports both single and batched graphs with memory-efficient sub-batching.
        
        Single graph (from MCTS/evaluation):
            - Input: graph_data with no 'batch'/'num_graphs' keys
            - Output: [1, output_dim] predictions
        
        Batched graphs (from training with sub-batching for memory efficiency):
            - Input: graph_data with 'batch' and 'num_graphs' keys
            - Divides into sub-batches for transformer (memory-efficient)
            - Concatenates embeddings from all sub-batches
            - Applies shared backbone and heads to full batch
            - Outputs: [num_graphs, output_dim] predictions
        
        Args:
            graph_data: Dict with node_features, edge_index, edge_weights, layer_positions
                       Plus optional: 'batch' tensor and 'num_graphs' count
            sub_batch_size: Number of graphs to process through transformer at once (default 4)
        
        Returns: {
            'action_type': [num_graphs, num_actions],
            'source_logits': [num_graphs, max_neurons],
            'target_logits': [num_graphs, max_neurons], 
            'activation_logits': [num_graphs, num_activations],
            'value': [num_graphs, 1],
        }
        """
        num_graphs = graph_data.get('num_graphs', 1)
        
        # Single graph case: process directly
        if num_graphs == 1:
            global_embedding, _ = self.graph_transformer(graph_data)
            # global_embedding: [1, hidden_dim * 2]
        elif sub_batch_size >= num_graphs:
            # Small batch case: no internal sub-batching needed
            # Trainer has already split into small sub-batches, so process entire sub-batch
            global_embedding, _ = self.graph_transformer(graph_data)
            # global_embedding: [1, hidden_dim * 2]
            # Now decompose into individual graphs
            global_embeddings = []
            batch_indices = graph_data['batch']  # [total_nodes]
            node_features = graph_data['node_features']  # [1, total_nodes, features]
            _, node_embeddings = self.graph_transformer(graph_data)
            # node_embeddings: [1, total_nodes, hidden_dim]
            node_embeddings_squeezed = node_embeddings.squeeze(0)  # [total_nodes, hidden_dim]
            
            # Pool embeddings per graph
            for graph_idx in range(num_graphs):
                mask = (batch_indices == graph_idx)
                if mask.any():
                    graph_embedding = node_embeddings_squeezed[mask].mean(dim=0)  # [hidden_dim]
                    global_embeddings.append(graph_embedding)
                else:
                    global_embeddings.append(torch.zeros(self.hidden_dim, device=node_embeddings.device, dtype=node_embeddings.dtype))
            
            # Stack embeddings: [num_graphs, hidden_dim]
            if global_embeddings:
                global_embedding = torch.stack(global_embeddings, dim=0)
                # Expand to match expected shape [num_graphs, hidden_dim * 2]
                global_embedding = torch.cat([global_embedding, global_embedding], dim=-1)  # [num_graphs, hidden_dim * 2]
            else:
                global_embedding = torch.zeros(num_graphs, self.hidden_dim * 2, device=node_features.device, dtype=node_features.dtype)
        else:
            # Large batch case: use internal sub-batching to avoid OOM in transformer attention
            # Key: Process sub-batches through transformer, but decompose into individual graphs
            global_embeddings = []
            batch_indices = graph_data['batch']  # [total_nodes]
            node_features = graph_data['node_features']  # [1, total_nodes, features]
            edge_index = graph_data['edge_index']
            edge_weights = graph_data['edge_weights']
            layer_positions = graph_data['layer_positions']  # [1, total_nodes]
            
            # Process graphs in sub-batches through transformer
            for sub_batch_start in range(0, num_graphs, sub_batch_size):
                sub_batch_end = min(sub_batch_start + sub_batch_size, num_graphs)
                graphs_in_sub_batch = list(range(sub_batch_start, sub_batch_end))
                
                # Create mask for nodes in this sub-batch
                sub_batch_mask = torch.zeros(len(batch_indices), dtype=torch.bool, device=batch_indices.device)
                for graph_idx in graphs_in_sub_batch:
                    sub_batch_mask |= (batch_indices == graph_idx)
                
                # Extract node indices for this sub-batch
                sub_batch_node_indices = torch.where(sub_batch_mask)[0]
                
                # Create mapping from old indices to new indices
                sub_batch_index_map = torch.full((len(batch_indices),), -1, dtype=torch.long, device=batch_indices.device)
                for new_idx, old_idx in enumerate(sub_batch_node_indices):
                    sub_batch_index_map[old_idx] = new_idx
                
                # Extract edges that are within this sub-batch
                valid_edges = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
                for edge_idx in range(edge_index.shape[1]):
                    src, tgt = edge_index[0, edge_idx], edge_index[1, edge_idx]
                    if sub_batch_index_map[src] == -1 or sub_batch_index_map[tgt] == -1:
                        valid_edges[edge_idx] = False
                
                # Remap edge indices for sub-batch
                sub_edge_index = edge_index[:, valid_edges].clone()
                for i in range(sub_edge_index.shape[1]):
                    sub_edge_index[0, i] = sub_batch_index_map[sub_edge_index[0, i]]
                    sub_edge_index[1, i] = sub_batch_index_map[sub_edge_index[1, i]]
                
                sub_edge_weights = edge_weights[valid_edges] if valid_edges.any() else edge_weights[:0]
                
                # Create sub-batch graph data (still contains multiple graphs)
                sub_batch_graph_data = {
                    'node_features': node_features[:, sub_batch_mask, :],  # [1, sub_batch_nodes, features]
                    'edge_index': sub_edge_index,
                    'edge_weights': sub_edge_weights,
                    'layer_positions': layer_positions[:, sub_batch_mask],  # [1, sub_batch_nodes]
                }
                
                # Now we need to decompose the sub-batch back into individual graphs
                # Get node embeddings from transformer to pool per-graph
                _, node_embeddings = self.graph_transformer(sub_batch_graph_data)
                # node_embeddings: [1, sub_batch_nodes, hidden_dim]
                node_embeddings_squeezed = node_embeddings.squeeze(0)  # [sub_batch_nodes, hidden_dim]
                
                # Create a mapping from sub-batch nodes back to graph indices
                sub_batch_node_to_graph = torch.full((len(sub_batch_node_indices),), -1, dtype=torch.long, device=batch_indices.device)
                for local_idx, global_idx in enumerate(sub_batch_node_indices):
                    graph_id = batch_indices[global_idx].item()
                    # Find position of this graph in graphs_in_sub_batch
                    pos_in_sub_batch = graphs_in_sub_batch.index(graph_id)
                    sub_batch_node_to_graph[local_idx] = pos_in_sub_batch
                
                # Pool embeddings per graph within this sub-batch
                num_graphs_in_sub_batch = len(graphs_in_sub_batch)
                for local_graph_idx in range(num_graphs_in_sub_batch):
                    mask = (sub_batch_node_to_graph == local_graph_idx)
                    if mask.any():
                        # Mean pool the node embeddings for this graph
                        graph_embedding = node_embeddings_squeezed[mask].mean(dim=0)  # [hidden_dim]
                        global_embeddings.append(graph_embedding)
                    else:
                        # Empty graph (shouldn't happen), use zeros
                        global_embeddings.append(torch.zeros(self.hidden_dim, device=node_embeddings.device, dtype=node_embeddings.dtype))
            
            # Stack embeddings: [num_graphs, hidden_dim]
            if global_embeddings:
                global_embedding = torch.stack(global_embeddings, dim=0)
                # Expand to match expected shape [num_graphs, hidden_dim * 2]
                # by concatenating with itself (simple approach) or creating proper dual embedding
                global_embedding = torch.cat([global_embedding, global_embedding], dim=-1)  # [num_graphs, hidden_dim * 2]
            else:
                global_embedding = torch.zeros(num_graphs, self.hidden_dim * 2, device=node_features.device, dtype=node_features.dtype)
        
        # Shared processing - applied to full batch (cheap linear ops, fits in memory)
        shared_features = self.shared_backbone(global_embedding)
        # shared_features: [num_graphs, hidden_dim // 2]
        
        # Policy heads
        action_type_logits = self.action_type_head(shared_features)
        source_logits = self.source_neuron_head(shared_features)
        target_logits = self.target_neuron_head(shared_features)
        activation_logits = self.activation_head(shared_features)
        
        # Value head
        value = self.value_head(shared_features)
        
        return {
            'action_type': action_type_logits,
            'source_logits': source_logits,
            'target_logits': target_logits,
            'activation_logits': activation_logits,
            'value': value,
        }

class ActionManager:
    """Manages action selection with masking and validation"""

    def __init__(self, max_neurons: int = 100, action_space: ActionSpace = None, exploration_boost: float = 0.5):
        self.max_neurons = max_neurons
        self.action_space = action_space
        self.exploration_boost = exploration_boost
        # Cache for masks to avoid recomputation
        self._mask_cache = {}
        self._cache_key_size = 0
    
    def get_action_masks(self, architecture: NeuralArchitecture) -> Dict[str, torch.Tensor]:
        """Get masks for invalid actions"""
        neurons = architecture.neurons
        
        # Optimized: filter hidden neurons once and cache neuron type info
        hidden_neurons = [nid for nid, neuron in neurons.items()
                         if neuron.neuron_type == NeuronType.HIDDEN]
        
        num_neurons = len(neurons)
        num_hidden = len(hidden_neurons)
        num_connections = len(architecture.connections)
        max_neuron_id = max(neurons.keys()) if neurons else 0
        tensor_size = max(max_neuron_id + 1, self.max_neurons)

        # Get the maximum neuron ID to determine tensor sizes
        # Create action mask with exploration incentives - vectorized
        action_mask = torch.full((5,), -1e9)  # 5 action types

        # Base validity checks (vectorized where possible)
        # We only set validity (0) or keep as heavily-penalized (-1e9).
        # Removed explicit exploration "buffs" so the policy head is not manually nudged here.
        if num_neurons < self.max_neurons:
            action_mask[ActionType.ADD_NEURON.value] = 0

        if num_hidden > 0:
            action_mask[ActionType.REMOVE_NEURON.value] = 0
            action_mask[ActionType.MODIFY_ACTIVATION.value] = 0

        if num_neurons >= 2:
            action_mask[ActionType.ADD_CONNECTION.value] = 0

        if num_connections > 0:
            action_mask[ActionType.REMOVE_CONNECTION.value] = 0

        # Source neuron mask (for remove/modify actions) - vectorized with indexing
        source_mask = torch.full((tensor_size,), -1e9)
        if hidden_neurons:
            valid_hidden = [nid for nid in hidden_neurons if nid < tensor_size]
            if valid_hidden:
                source_mask[valid_hidden] = 0

        # Target neuron mask (for connection actions) - vectorized with indexing
        target_mask = torch.full((tensor_size,), -1e9)
        valid_neuron_ids = [nid for nid in neurons.keys() if nid < tensor_size]
        if valid_neuron_ids:
            target_mask[valid_neuron_ids] = 0

        return {
            'action_type': action_mask,
            'source_neurons': source_mask,
            'target_neurons': target_mask,
            'tensor_size': tensor_size
        }
    
    def _prepare_logits(self, logits_full: torch.Tensor, tensor_size: int, mask: torch.Tensor) -> torch.Tensor:
        """Helper to pad/truncate logits and apply mask - optimized"""
        device = logits_full.device
        is_batched = logits_full.dim() > 1
        
        if is_batched:
            batch_size = logits_full.shape[0]
            current_size = logits_full.shape[1]
        else:
            batch_size = 1
            current_size = logits_full.shape[0]
        
        # Only pad if necessary
        if current_size < tensor_size:
            padding_size = tensor_size - current_size
            if is_batched:
                padding = torch.full((batch_size, padding_size), -1e9, device=device)
                logits = torch.cat([logits_full, padding], dim=1)
            else:
                padding = torch.full((padding_size,), -1e9, device=device)
                logits = torch.cat([logits_full, padding])
        else:
            logits = logits_full[:, :tensor_size] if is_batched else logits_full[:tensor_size]
        
        # Apply mask (broadcast over batch if needed)
        return logits + (mask.unsqueeze(0) if is_batched else mask)
    
    def select_action(self, policy_output: Dict, architecture: NeuralArchitecture,
                     exploration: bool = True, use_policy: bool = True) -> Action:
        """Select action using policy output with masking, or random if use_policy=False"""
        if not use_policy:
            # Use random action selection from action space
            if self.action_space is None:
                raise ValueError("ActionSpace must be provided to ActionManager for random action selection")
            valid_actions = self.action_space.get_valid_actions(architecture)
            if valid_actions:
                return np.random.choice(valid_actions)
            else:
                raise ValueError("No valid actions available in the current architecture")

        masks_dict = self.get_action_masks(architecture)
        masks = masks_dict.copy()
        tensor_size = masks.pop('tensor_size')
        
        # Move masks to the same device as policy_output
        device = policy_output['action_type'].device
        for key in masks:
            masks[key] = masks[key].to(device)

        # Sample action type
        action_logits = policy_output['action_type'] + masks['action_type']

        if exploration:
            action_type_idx = Categorical(logits=action_logits).sample().item()
        else:
            action_type_idx = action_logits.argmax().item()

        action_type = ActionType(action_type_idx)
        
        # Select parameters based on action type - optimized with early returns
        if action_type == ActionType.ADD_NEURON:
            activation_logits = policy_output['activation_logits']
            if exploration:
                activation_idx = Categorical(logits=activation_logits).sample().item()
            else:
                activation_idx = activation_logits.argmax().item()
            
            return Action(
                action_type=action_type,
                activation=list(ActivationType)[activation_idx % len(ActivationType)]
            )
        
        if action_type == ActionType.REMOVE_NEURON:
            source_logits = self._prepare_logits(
                policy_output['source_logits'], tensor_size, masks['source_neurons']
            )
            
            if exploration:
                source_idx = Categorical(logits=source_logits).sample().item()
            else:
                source_idx = source_logits.argmax().item()
            
            return Action(
                action_type=action_type,
                source_neuron=source_idx
            )
        
        if action_type == ActionType.MODIFY_ACTIVATION:
            source_logits = self._prepare_logits(
                policy_output['source_logits'], tensor_size, masks['source_neurons']
            )
            activation_logits = policy_output['activation_logits']
            
            if exploration:
                source_idx = Categorical(logits=source_logits).sample().item()
                activation_idx = Categorical(logits=activation_logits).sample().item()
            else:
                source_idx = source_logits.argmax().item()
                activation_idx = activation_logits.argmax().item()
            
            return Action(
                action_type=action_type,
                source_neuron=source_idx,
                activation=list(ActivationType)[activation_idx % len(ActivationType)]
            )
        
        if action_type == ActionType.ADD_CONNECTION:
            # Use helper to prepare logits efficiently
            # Prepare source logits using the source_neurons mask (was incorrectly using target_neurons)
            source_logits = self._prepare_logits(
                policy_output['source_logits'], tensor_size, masks['source_neurons']
            )
            target_logits = self._prepare_logits(
                policy_output['target_logits'], tensor_size, masks['target_neurons']
            )
            
            # Sample source first
            if exploration:
                source_idx = Categorical(logits=source_logits).sample().item()
            else:
                source_idx = source_logits.argmax().item()

            # Prevent self-connection by penalizing the chosen source index in target logits
            if 0 <= source_idx < tensor_size:
                if target_logits.dim() == 1:
                    target_logits[source_idx] = -1e9
                else:
                    target_logits[:, source_idx] = -1e9

            if exploration:
                target_idx = Categorical(logits=target_logits).sample().item()
            else:
                target_idx = target_logits.argmax().item()

            return Action(
                action_type=action_type,
                source_neuron=source_idx,
                target_neuron=target_idx
            )
        
        # REMOVE_CONNECTION case
        existing_connections = architecture.connections
        if not existing_connections:
            # Fallback to add connection if no connections to remove
            return self.select_action(policy_output, architecture, exploration)

        # Extract and flatten logits if batched - optimized with single pass
        source_logits_full = policy_output['source_logits']
        target_logits_full = policy_output['target_logits']
        
        if source_logits_full.dim() > 1:
            source_logits_full = source_logits_full[0]
        if target_logits_full.dim() > 1:
            target_logits_full = target_logits_full[0]

        device = source_logits_full.device
        
        # Pad to tensor_size if needed - optimized
        if source_logits_full.shape[0] < tensor_size:
            source_logits_full = torch.cat([
                source_logits_full, 
                torch.full((tensor_size - source_logits_full.shape[0],), -1e9, device=device)
            ])
        else:
            source_logits_full = source_logits_full[:tensor_size]
            
        if target_logits_full.shape[0] < tensor_size:
            target_logits_full = torch.cat([
                target_logits_full,
                torch.full((tensor_size - target_logits_full.shape[0],), -1e9, device=device)
            ])
        else:
            target_logits_full = target_logits_full[:tensor_size]

        # Vectorized scoring: build tensors of source/target IDs, then index in parallel
        source_ids = torch.tensor([conn.source_id for conn in existing_connections], device=device)
        target_ids = torch.tensor([conn.target_id for conn in existing_connections], device=device)
        
        # Clamp indices to valid range to prevent indexing errors
        source_ids = source_ids.clamp(0, tensor_size - 1)
        target_ids = target_ids.clamp(0, tensor_size - 1)
        
        # Vectorized lookup and sum
        combined_logits = source_logits_full[source_ids] + target_logits_full[target_ids]

        if exploration:
            sel_idx = Categorical(logits=combined_logits).sample().item()
        else:
            sel_idx = combined_logits.argmax().item()

        sel_conn = existing_connections[sel_idx]
        return Action(
            action_type=action_type,
            source_neuron=sel_conn.source_id,
            target_neuron=sel_conn.target_id
        )