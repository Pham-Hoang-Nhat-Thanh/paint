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
        
        # Shared backbone MLP - optimized for speed
        self.shared_backbone = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),  # global_embedding is hidden_dim * 2
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Policy heads (factorized action space) - optimized
        self.action_type_head = nn.Linear(hidden_dim // 2, num_actions, bias=False)
        self.source_neuron_head = nn.Linear(hidden_dim // 2, max_neurons, bias=False)
        self.target_neuron_head = nn.Linear(hidden_dim // 2, max_neurons, bias=False)
        self.activation_head = nn.Linear(hidden_dim // 2, num_activations, bias=False)
        
        # Value head - optimized
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=False),
            nn.ReLU(),
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
        hidden_neurons = [nid for nid, neuron in neurons.items()
                         if neuron.neuron_type == NeuronType.HIDDEN]

        # Get the maximum neuron ID to determine tensor sizes
        max_neuron_id = max(neurons.keys()) if neurons else 0
        tensor_size = max(max_neuron_id + 1, self.max_neurons)

        # Action type mask with exploration incentives
        action_mask = torch.zeros(5) - 1e9  # 5 action types

        # Base validity
        if len(neurons) < self.max_neurons:
            action_mask[ActionType.ADD_NEURON.value] = 0

        if hidden_neurons:
            action_mask[ActionType.REMOVE_NEURON.value] = 0
            action_mask[ActionType.MODIFY_ACTIVATION.value] = 0

        if len(neurons) >= 2:
            action_mask[ActionType.ADD_CONNECTION.value] = 0

        if architecture.connections:
            action_mask[ActionType.REMOVE_CONNECTION.value] = 0

        # Exploration incentives: slightly boost underrepresented action types
        num_neurons = len(neurons)
        num_connections = len(architecture.connections)

        # Boost neuron actions if architecture is too small
        if num_neurons < 10:
            if action_mask[ActionType.ADD_NEURON.value] == 0:
                action_mask[ActionType.ADD_NEURON.value] += self.exploration_boost

        # Boost connection actions if architecture has few connections relative to neurons
        if num_connections < num_neurons * 2 and num_neurons >= 3:
            if action_mask[ActionType.ADD_CONNECTION.value] == 0:
                action_mask[ActionType.ADD_CONNECTION.value] += self.exploration_boost * 0.6

        # Boost modification actions if architecture is stable
        if num_connections > num_neurons and hidden_neurons:
            if action_mask[ActionType.MODIFY_ACTIVATION.value] == 0:
                action_mask[ActionType.MODIFY_ACTIVATION.value] += self.exploration_boost * 0.4

        masks['action_type'] = action_mask

        # Source neuron mask (for remove/modify actions)
        source_mask = torch.zeros(tensor_size) - 1e9
        for neuron_id in hidden_neurons:
            if neuron_id < tensor_size:  # Ensure within bounds
                source_mask[neuron_id] = 0
        masks['source_neurons'] = source_mask

        # Target neuron mask (for connection actions)
        target_mask = torch.zeros(tensor_size) - 1e9
        for neuron_id in neurons.keys():
            if neuron_id < tensor_size:  # Ensure within bounds
                target_mask[neuron_id] = 0
        masks['target_neurons'] = target_mask

        return masks
    
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
            # Get the relevant portion of source logits (handle [features] or [batch, features])
            source_logits_full = policy_output['source_logits']
            # Ensure we operate on the feature dimension (last dim)
            if source_logits_full.dim() == 1:
                if source_logits_full.shape[0] < tensor_size:
                    padding = torch.full((tensor_size - source_logits_full.shape[0],), -1e9, device=source_logits_full.device)
                    source_logits = torch.cat([source_logits_full, padding])
                else:
                    source_logits = source_logits_full[:tensor_size]
                # masks is 1D
                source_logits = source_logits + masks['source_neurons'].to(source_logits.device)
            else:
                # [batch, features]
                batch_size = source_logits_full.shape[0]
                if source_logits_full.shape[1] < tensor_size:
                    padding = torch.full((batch_size, tensor_size - source_logits_full.shape[1]), -1e9, device=source_logits_full.device)
                    source_logits = torch.cat([source_logits_full, padding], dim=1)
                else:
                    source_logits = source_logits_full[:, :tensor_size]
                # broadcast mask over batch
                source_logits = source_logits + masks['source_neurons'].to(source_logits.device).unsqueeze(0)
            
            if exploration:
                source_idx = Categorical(logits=source_logits).sample().item()
            else:
                source_idx = source_logits.argmax().item()
            
            return Action(
                action_type=action_type,
                source_neuron=source_idx
            )
        
        elif action_type == ActionType.MODIFY_ACTIVATION:
            # Get the relevant portion of source logits (handle [features] or [batch, features])
            source_logits_full = policy_output['source_logits']
            if source_logits_full.dim() == 1:
                if source_logits_full.shape[0] < tensor_size:
                    padding = torch.full((tensor_size - source_logits_full.shape[0],), -1e9, device=source_logits_full.device)
                    source_logits = torch.cat([source_logits_full, padding])
                else:
                    source_logits = source_logits_full[:tensor_size]
                source_logits = source_logits + masks['source_neurons'].to(source_logits.device)
            else:
                batch_size = source_logits_full.shape[0]
                if source_logits_full.shape[1] < tensor_size:
                    padding = torch.full((batch_size, tensor_size - source_logits_full.shape[1]), -1e9, device=source_logits_full.device)
                    source_logits = torch.cat([source_logits_full, padding], dim=1)
                else:
                    source_logits = source_logits_full[:, :tensor_size]
                source_logits = source_logits + masks['source_neurons'].to(source_logits.device).unsqueeze(0)
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
            # Get the relevant portions of source and target logits (handle 1D or 2D)
            source_logits_full = policy_output['source_logits']
            target_logits_full = policy_output['target_logits']
            
            # Handle source logits
            if source_logits_full.dim() == 1:
                if source_logits_full.shape[0] < tensor_size:
                    padding = torch.full((tensor_size - source_logits_full.shape[0],), -1e9, device=source_logits_full.device)
                    source_logits = torch.cat([source_logits_full, padding])
                else:
                    source_logits = source_logits_full[:tensor_size]
                source_logits = source_logits + masks['target_neurons'].to(source_logits.device)
            else:
                batch_size = source_logits_full.shape[0]
                if source_logits_full.shape[1] < tensor_size:
                    padding = torch.full((batch_size, tensor_size - source_logits_full.shape[1]), -1e9, device=source_logits_full.device)
                    source_logits = torch.cat([source_logits_full, padding], dim=1)
                else:
                    source_logits = source_logits_full[:, :tensor_size]
                source_logits = source_logits + masks['target_neurons'].to(source_logits.device).unsqueeze(0)

            # Handle target logits
            if target_logits_full.dim() == 1:
                if target_logits_full.shape[0] < tensor_size:
                    padding = torch.full((tensor_size - target_logits_full.shape[0],), -1e9, device=target_logits_full.device)
                    target_logits = torch.cat([target_logits_full, padding])
                else:
                    target_logits = target_logits_full[:tensor_size]
                target_logits = target_logits + masks['target_neurons'].to(target_logits.device)
            else:
                batch_size = target_logits_full.shape[0]
                if target_logits_full.shape[1] < tensor_size:
                    padding = torch.full((batch_size, tensor_size - target_logits_full.shape[1]), -1e9, device=target_logits_full.device)
                    target_logits = torch.cat([target_logits_full, padding], dim=1)
                else:
                    target_logits = target_logits_full[:, :tensor_size]
                target_logits = target_logits + masks['target_neurons'].to(target_logits.device).unsqueeze(0)
            
            # Sample source first, then mask only the corresponding target index
            # (prevents globally penalizing all targets and allows diverse connections)
            if exploration:
                source_idx = Categorical(logits=source_logits).sample().item()
            else:
                source_idx = source_logits.argmax().item()

            # Prevent self-connection by penalizing the chosen source index in target logits
            # target_logits may be 1D ([features]) or 2D ([batch, features]).
            try:
                if target_logits.dim() == 1:
                    if 0 <= source_idx < target_logits.shape[0]:
                        target_logits[source_idx] -= 1e9
                else:
                    if 0 <= source_idx < target_logits.shape[1]:
                        target_logits[:, source_idx] -= 1e9
            except Exception:
                traceback.print_exc()
                # Defensive: if shapes are unexpected, fall back to masking all indices
                if target_logits.dim() == 1:
                    target_logits[:tensor_size] -= 1e9
                else:
                    target_logits[:, :tensor_size] -= 1e9

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
            # Select a connection to remove using the learned source/target logits.
            existing_connections = list(architecture.connections)
            if not existing_connections:
                # Fallback to add connection if no connections to remove
                return self.select_action(policy_output, architecture, exploration)

            # Prepare 1D source and target logits (prefer first batch row if batched)
            source_logits_full = policy_output.get('source_logits')
            target_logits_full = policy_output.get('target_logits')

            # If batched, use the first entry as a proxy for scoring
            if source_logits_full.dim() > 1:
                source_logits_full = source_logits_full[0]
            if target_logits_full.dim() > 1:
                target_logits_full = target_logits_full[0]

            # Pad/truncate to tensor_size, using 0.0 for padding since we don't mask for remove
            if source_logits_full.shape[0] < tensor_size:
                pad = torch.full((tensor_size - source_logits_full.shape[0],), 0.0, device=source_logits_full.device)
                source_logits = torch.cat([source_logits_full, pad])
            else:
                source_logits = source_logits_full[:tensor_size]

            if target_logits_full.shape[0] < tensor_size:
                pad = torch.full((tensor_size - target_logits_full.shape[0],), 0.0, device=target_logits_full.device)
                target_logits = torch.cat([target_logits_full, pad])
            else:
                target_logits = target_logits_full[:tensor_size]

            # Build combined logits for each existing connection (source_logit + target_logit)
            combined = []
            for conn in existing_connections:
                s_id = conn.source_id
                t_id = conn.target_id
                # Defensive bounds check
                if s_id < 0 or s_id >= source_logits.shape[0] or t_id < 0 or t_id >= target_logits.shape[0]:
                    combined.append(-1e9)
                else:
                    combined.append((source_logits[s_id] + target_logits[t_id]).item())

            combined_logits = torch.tensor(combined, device=source_logits.device)

            if exploration:
                try:
                    sel_idx = Categorical(logits=combined_logits).sample().item()
                except Exception:
                    traceback.print_exc()
                    # Fallback: pick highest-scoring connection
                    sel_idx = int(combined_logits.argmax().item())
            else:
                sel_idx = int(combined_logits.argmax().item())

            sel_conn = existing_connections[sel_idx]
            return Action(
                action_type=action_type,
                source_neuron=sel_conn.source_id,
                target_neuron=sel_conn.target_id
            )
        
        else:
            raise ValueError(f"Unknown action type: {action_type}")