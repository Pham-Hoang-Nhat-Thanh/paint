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
    
    def forward(self, graph_data: Dict) -> Dict[str, torch.Tensor]:
        """
        Returns: {
            'action_type': [batch_size, num_actions],
            'source_logits': [batch_size, max_neurons],
            'target_logits': [batch_size, max_neurons], 
            'activation_logits': [batch_size, num_activations],
            'value': [batch_size, 1],
            'global_embedding': [batch_size, hidden_dim * 2],
            'node_embeddings': [batch_size, num_nodes, hidden_dim]
        }
        """
        # Get graph embeddings
        global_embedding, node_embeddings = self.graph_transformer(graph_data)
        
        # Shared processing
        shared_features = self.shared_backbone(global_embedding)
        
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
            'global_embedding': global_embedding,
            'node_embeddings': node_embeddings
        }

class ActionManager:
    """Manages action selection with masking and validation"""

    def __init__(self, max_neurons: int = 100, action_space: ActionSpace = None, exploration_boost: float = 0.5):
        self.max_neurons = max_neurons
        self.action_space = action_space
        self.exploration_boost = exploration_boost
    
    def get_action_masks(self, architecture: NeuralArchitecture) -> Dict[str, torch.Tensor]:
        """Get masks for invalid actions with balanced exploration incentives"""
        masks = {}
        neurons = architecture.neurons
        
        # Optimized: filter hidden neurons without repeated dictionary lookups
        hidden_neurons = [nid for nid, neuron in neurons.items()
                         if neuron.neuron_type == NeuronType.HIDDEN]
        
        num_neurons = len(neurons)
        num_hidden = len(hidden_neurons)
        num_connections = len(architecture.connections)

        # Get the maximum neuron ID to determine tensor sizes
        max_neuron_id = max(neurons.keys()) if neurons else 0
        tensor_size = max(max_neuron_id + 1, self.max_neurons)

        # Action type mask with exploration incentives - optimized with vectorized operations
        action_mask = torch.full((5,), -1e9)  # 5 action types

        # Base validity checks (vectorized where possible)
        if num_neurons < self.max_neurons:
            action_mask[ActionType.ADD_NEURON.value] = 0
            # Boost neuron actions if architecture is too small
            if num_neurons < 10:
                action_mask[ActionType.ADD_NEURON.value] += self.exploration_boost

        if num_hidden > 0:
            action_mask[ActionType.REMOVE_NEURON.value] = 0
            action_mask[ActionType.MODIFY_ACTIVATION.value] = 0
            # Boost modification actions if architecture is stable
            if num_connections > num_neurons:
                action_mask[ActionType.MODIFY_ACTIVATION.value] += self.exploration_boost * 0.4

        if num_neurons >= 2:
            action_mask[ActionType.ADD_CONNECTION.value] = 0
            # Boost connection actions if architecture has few connections relative to neurons
            if num_connections < num_neurons * 2 and num_neurons >= 3:
                action_mask[ActionType.ADD_CONNECTION.value] += self.exploration_boost * 0.6

        if num_connections > 0:
            action_mask[ActionType.REMOVE_CONNECTION.value] = 0

        masks['action_type'] = action_mask

        # Source neuron mask (for remove/modify actions) - optimized with vectorized indexing
        source_mask = torch.full((tensor_size,), -1e9)
        if hidden_neurons:
            valid_hidden = [nid for nid in hidden_neurons if nid < tensor_size]
            if valid_hidden:
                source_mask[valid_hidden] = 0
        masks['source_neurons'] = source_mask

        # Target neuron mask (for connection actions) - optimized with vectorized indexing
        target_mask = torch.full((tensor_size,), -1e9)
        valid_neuron_ids = [nid for nid in neurons.keys() if nid < tensor_size]
        if valid_neuron_ids:
            target_mask[valid_neuron_ids] = 0
        masks['target_neurons'] = target_mask

        return masks
    
    def _prepare_logits(self, logits_full: torch.Tensor, tensor_size: int, mask: torch.Tensor) -> torch.Tensor:
        """Helper to pad/truncate logits and apply mask - optimized"""
        device = logits_full.device
        is_batched = logits_full.dim() > 1
        
        if is_batched:
            batch_size = logits_full.shape[0]
            current_size = logits_full.shape[1]
            if current_size < tensor_size:
                padding = torch.full((batch_size, tensor_size - current_size), -1e9, device=device)
                logits = torch.cat([logits_full, padding], dim=1)
            else:
                logits = logits_full[:, :tensor_size]
            # Broadcast mask over batch
            return logits + mask.unsqueeze(0)
        else:
            current_size = logits_full.shape[0]
            if current_size < tensor_size:
                padding = torch.full((tensor_size - current_size,), -1e9, device=device)
                logits = torch.cat([logits_full, padding])
            else:
                logits = logits_full[:tensor_size]
            return logits + mask
    
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

        masks = self.get_action_masks(architecture)
        # Move masks to the same device as policy_output
        device = policy_output['action_type'].device
        for key in masks:
            masks[key] = masks[key].to(device)

        # Get the actual tensor size needed
        max_neuron_id = max(architecture.neurons.keys()) if architecture.neurons else 0
        tensor_size = max(max_neuron_id + 1, self.max_neurons)

        # Sample action type
        action_logits = policy_output['action_type'] + masks['action_type']

        if exploration:
            action_type_idx = Categorical(logits=action_logits).sample().item()
        else:
            action_type_idx = action_logits.argmax().item()

        action_type = ActionType(action_type_idx)
        
        # Select parameters based on action type
        if action_type == ActionType.ADD_NEURON:
            activation_logits = policy_output['activation_logits']
            if exploration:
                activation_idx = Categorical(logits=activation_logits).sample().item()
            else:
                activation_idx = activation_logits.argmax().item()
            
            return Action(
                action_type=action_type,
                activation=list(ActivationType)[activation_idx % 4]
            )
        
        elif action_type == ActionType.REMOVE_NEURON:
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
        
        elif action_type == ActionType.MODIFY_ACTIVATION:
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
                activation=list(ActivationType)[activation_idx % 4]
            )
        
        elif action_type == ActionType.ADD_CONNECTION:
            # Use helper to prepare logits efficiently
            source_logits = self._prepare_logits(
                policy_output['source_logits'], tensor_size, masks['target_neurons']
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
        
        elif action_type == ActionType.REMOVE_CONNECTION:
            existing_connections = architecture.connections
            if not existing_connections:
                # Fallback to add connection if no connections to remove
                return self.select_action(policy_output, architecture, exploration)

            # Extract and flatten logits if batched
            source_logits_full = policy_output['source_logits']
            target_logits_full = policy_output['target_logits']
            
            if source_logits_full.dim() > 1:
                source_logits_full = source_logits_full[0]
            if target_logits_full.dim() > 1:
                target_logits_full = target_logits_full[0]

            device = source_logits_full.device
            
            # Pad to tensor_size if needed
            if source_logits_full.shape[0] < tensor_size:
                source_logits_full = torch.cat([
                    source_logits_full, 
                    torch.zeros(tensor_size - source_logits_full.shape[0], device=device)
                ])
            if target_logits_full.shape[0] < tensor_size:
                target_logits_full = torch.cat([
                    target_logits_full,
                    torch.zeros(tensor_size - target_logits_full.shape[0], device=device)
                ])

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
        
        else:
            raise ValueError(f"Unknown action type: {action_type}")