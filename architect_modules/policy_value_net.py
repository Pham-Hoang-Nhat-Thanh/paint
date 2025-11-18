import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from .graph_transformer import GraphTransformer
import traceback
from typing import Dict, List
from blueprint_modules.network import NeuralArchitecture, NeuronType, ActivationType
from blueprint_modules.action import ActionType, Action, ActionSpace

# Enable TF32 for faster matrix multiplications on Ampere+ GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class UnifiedPolicyValueNetwork(nn.Module):
    """Unified network that outputs both policy and value predictions"""
    
    def __init__(self, node_feature_dim: int, hidden_dim: int = 128,
                 max_neurons: int = 1000, num_actions: int = 5,
                 num_activations: int = 4, self_attention_heads: int = 8,
                 transformer_layers: int = 3, num_phases: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_neurons = max_neurons
        self.num_actions = num_actions
        self.num_activations = num_activations

        # Phase embedding
        self.phase_embedding = nn.Embedding(num_phases, hidden_dim // 8)
        
        # Graph transformer backbone
        self.graph_transformer = GraphTransformer(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=self_attention_heads,
            num_layers=transformer_layers
        )
        
        # Shared backbone - now includes action type conditioning
        self.shared_backbone = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 8, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # Action type head (unconditional)
        self.action_type_head = nn.Linear(hidden_dim // 2, num_actions, bias=False)
        
        # Conditional source heads - one per action type that needs sources
        self.conditional_source_heads = nn.ModuleDict({
            'remove_neuron': nn.Linear(hidden_dim // 2, max_neurons, bias=False),
            'add_connection': nn.Linear(hidden_dim // 2, max_neurons, bias=False),
            'remove_connection': nn.Linear(hidden_dim // 2, max_neurons, bias=False),
            'modify_activation': nn.Linear(hidden_dim // 2, max_neurons, bias=False)
        })
        
        # Conditional target heads - conditioned on source features
        self.conditional_target_heads = nn.ModuleDict({
            'add_connection': nn.Sequential(
                nn.Linear(hidden_dim // 2 + hidden_dim // 8, hidden_dim // 4, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 4, max_neurons, bias=False)
            ),
            'remove_connection': nn.Sequential(
                nn.Linear(hidden_dim // 2 + hidden_dim // 8, hidden_dim // 4, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 4, max_neurons, bias=False)
            )
        })
        
        # Conditional activation heads
        self.conditional_activation_heads = nn.ModuleDict({
            'add_neuron': nn.Linear(hidden_dim // 2, num_activations, bias=False),
            'modify_activation': nn.Sequential(
                nn.Linear(hidden_dim // 2 + hidden_dim // 8, hidden_dim // 4, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 4, num_activations, bias=False)
            )
        })
        
        # Source feature encoder (for conditioning target/activation heads on source)
        self.source_encoder = nn.Sequential(
            nn.Linear(max_neurons, hidden_dim // 8, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # Value head
        self.value_head = nn.Linear(hidden_dim // 2, 1, bias=False)

        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, graph_data: Dict, phase: int) -> Dict[str, torch.Tensor]:
        num_graphs = graph_data.get('num_graphs', 1)
        model_device = next(self.parameters()).device
        
        # Ensure all tensors are on correct device
        for k, v in list(graph_data.items()):
            if torch.is_tensor(v) and v.device != model_device:
                graph_data[k] = v.to(model_device)
        
        # Get global embedding
        if num_graphs == 1:
            global_embedding, _ = self.graph_transformer(graph_data)
        else:
            global_embedding, node_embeddings = self.graph_transformer(graph_data)
            batch_indices = graph_data['batch']
            node_embeddings_squeezed = node_embeddings.squeeze(0)
            
            device = batch_indices.device
            batch_idx = batch_indices.long()
            total_nodes, hidden_dim = node_embeddings_squeezed.shape
            
            pooled = torch.zeros((num_graphs, hidden_dim), device=device)
            pooled.index_add_(0, batch_idx, node_embeddings_squeezed)
            
            counts = torch.zeros((num_graphs,), device=device)
            counts.index_add_(0, batch_idx, torch.ones((total_nodes,), device=device))
            counts = counts.clamp(min=1).unsqueeze(-1)
            
            pooled_embeddings = pooled / counts
            global_embedding = torch.cat([pooled_embeddings, pooled_embeddings], dim=-1)
        
        # Add phase embedding
        phase_tensor = torch.tensor([phase] * num_graphs, dtype=torch.long, device=model_device)
        phase_embedding = self.phase_embedding(phase_tensor)
        global_embedding = torch.cat([global_embedding, phase_embedding], dim=-1)
        
        # Shared processing
        shared_features = self.shared_backbone(global_embedding)
        
        # Action type (unconditional)
        action_type_logits = self.action_type_head(shared_features)
        
        # Conditional source logits for each action type
        source_logits_dict = {}
        for action_name, head in self.conditional_source_heads.items():
            source_logits_dict[action_name] = head(shared_features)
        
        # For target and activation, we return the networks to compute them conditionally
        # during action selection when we know the source
        return {
            'action_type': action_type_logits,
            'source_logits_dict': source_logits_dict,
            'target_heads': self.conditional_target_heads,
            'activation_heads': self.conditional_activation_heads,
            'shared_features': shared_features,
            'source_encoder': self.source_encoder,
            'value': self.value_head(shared_features)
        }
class ActionManager:
    """Manages action selection with masking, validation, and conditional distributions"""

    def __init__(self, max_neurons: int = 100, action_space: ActionSpace = None, exploration_boost: float = 0.5):
        self.max_neurons = max_neurons
        self.action_space = action_space
        self.exploration_boost = exploration_boost
        # Cache for masks to avoid recomputation
        self._mask_cache = {}
        self._cache_key_size = 0

    def _get_action_type_name(self, action_type: int) -> str:
        """Convert action type index to string name"""
        action_names = {
            0: 'add_neuron',
            1: 'remove_neuron', 
            2: 'add_connection',
            3: 'remove_connection',
            4: 'modify_activation'
        }
        return action_names.get(action_type, 'add_neuron')
    
    def _compute_priors_vectorized(self, policy_output: Dict, actions: List[Action], masks: Dict) -> torch.Tensor:
        """Compute priors using correct conditional probabilities in vectorized form"""
        if not actions:
            return torch.tensor([], device=next(iter(policy_output.values())).device)

        device = policy_output['action_type'].device
        n_actions = len(actions)
        priors = torch.ones(n_actions, device=device)

        # Precompute action type probabilities (squeezed)
        action_type_logits = policy_output['action_type'].squeeze(0)
        masked_action_logits = action_type_logits + masks['action_type'].to(device)
        action_type_probs = F.softmax(masked_action_logits, dim=-1)

        # Cache conditional logits keyed by (action_type_value, source_neuron_or_-1)
        cond_cache = {}

        # Helper to normalize logits safely (handles batched/unbatched forms)
        def _softmax_prep(logits_tensor, mask_tensor):
            # logits_tensor may be 1D (squeezed) or 2D (batch x size)
            if logits_tensor.dim() == 1:
                l_full = logits_tensor.unsqueeze(0)
            else:
                l_full = logits_tensor
            l_prep = self._prepare_logits(l_full, masks['tensor_size'], mask_tensor.to(device))
            # return 1D probabilities
            if l_prep.dim() > 1:
                return F.softmax(l_prep.squeeze(0), dim=-1)
            return F.softmax(l_prep, dim=-1)

        # Process each action but reuse conditional computations when possible
        for i, action in enumerate(actions):
            at_val = action.action_type.value
            prior = action_type_probs[at_val]

            src_key = action.source_neuron if action.source_neuron is not None else -1
            cache_key = (at_val, int(src_key))

            if cache_key not in cond_cache:
                # compute and store conditional logits with batch-dim squeezed where safe
                cond_logits = self._compute_conditional_logits(policy_output, action.action_type, action.source_neuron)
                for k, v in list(cond_logits.items()):
                    if torch.is_tensor(v):
                        v = v.to(device)
                        # Squeeze leading batch dimension if it's a singleton batch
                        if v.dim() > 1 and v.shape[0] == 1:
                            v = v.squeeze(0)
                        cond_logits[k] = v
                cond_cache[cache_key] = cond_logits

            cond = cond_cache[cache_key]

            if action.action_type == ActionType.ADD_NEURON:
                activation_probs = F.softmax(cond['activation_logits'], dim=-1)
                activation_idx = list(ActivationType).index(action.activation)
                prior = prior * activation_probs[activation_idx]

            elif action.action_type == ActionType.REMOVE_NEURON:
                sp = _softmax_prep(cond['source_logits'], masks.get('remove_source_neurons', masks['source_neurons']))
                prior = prior * sp[action.source_neuron]

            elif action.action_type in [ActionType.ADD_CONNECTION, ActionType.REMOVE_CONNECTION]:
                sp = _softmax_prep(cond['source_logits'], masks['source_neurons'])

                # For target, the cached cond logits may already be conditioned on source (if computed with source),
                # otherwise it's the unconditioned head and _softmax_prep will pad/softmax correctly.
                tp = _softmax_prep(cond['target_logits'], masks['target_neurons'])
                prior = prior * sp[action.source_neuron] * tp[action.target_neuron]

            elif action.action_type == ActionType.MODIFY_ACTIVATION:
                sp = _softmax_prep(cond['source_logits'], masks.get('modify_source_neurons', masks['source_neurons']))
                activation_probs = F.softmax(cond['activation_logits'], dim=-1)
                activation_idx = list(ActivationType).index(action.activation)
                prior = prior * sp[action.source_neuron] * activation_probs[activation_idx]

            priors[i] = prior

        return priors

    def _compute_conditional_logits(self, policy_output: Dict, action_type: ActionType, 
                                  source_neuron: int = None) -> Dict[str, torch.Tensor]:
        """Compute conditional logits given action type and optional source"""
        shared_features = policy_output['shared_features']
        action_type_name = self._get_action_type_name(action_type.value)
        
        result = {}
        
        # Source logits (conditioned on action type)
        if 'source_logits_dict' in policy_output and action_type_name in policy_output['source_logits_dict']:
            result['source_logits'] = policy_output['source_logits_dict'][action_type_name]
        else:
            # Fallback to original source logits for backward compatibility
            result['source_logits'] = policy_output.get('source_logits', 
                                                       torch.zeros(shared_features.shape[0], self.max_neurons, 
                                                                  device=shared_features.device))
        
        # Target logits (conditioned on action type and source)
        if (source_neuron is not None and 'target_heads' in policy_output and 
            action_type_name in policy_output['target_heads']):
            # Encode source information
            source_one_hot = F.one_hot(torch.tensor([source_neuron], device=shared_features.device), 
                                      self.max_neurons).float()
            source_features = policy_output['source_encoder'](source_one_hot)
            
            # Combine with shared features
            conditioned_features = torch.cat([shared_features, source_features], dim=-1)
            result['target_logits'] = policy_output['target_heads'][action_type_name](conditioned_features)
        else:
            # Fallback to original target logits
            result['target_logits'] = policy_output.get('target_logits', 
                                                       torch.zeros(shared_features.shape[0], self.max_neurons,
                                                                  device=shared_features.device))
        
        # Activation logits (conditioned on action type and source if needed)
        if 'activation_heads' in policy_output and action_type_name in policy_output['activation_heads']:
            if action_type_name == 'modify_activation' and source_neuron is not None:
                # Condition on source for modify_activation
                source_one_hot = F.one_hot(torch.tensor([source_neuron], device=shared_features.device), 
                                          self.max_neurons).float()
                source_features = policy_output['source_encoder'](source_one_hot)
                conditioned_features = torch.cat([shared_features, source_features], dim=-1)
                result['activation_logits'] = policy_output['activation_heads'][action_type_name](conditioned_features)
            else:
                # Unconditional for add_neuron
                result['activation_logits'] = policy_output['activation_heads'][action_type_name](shared_features)
        else:
            # Fallback to original activation logits
            result['activation_logits'] = policy_output.get('activation_logits', 
                                                           torch.zeros(shared_features.shape[0], 4, 
                                                                      device=shared_features.device))
        
        return result
    
    def get_action_masks(self, architecture: NeuralArchitecture, phase: int) -> Dict[str, torch.Tensor]:
        """Get masks for invalid actions"""
        neurons = architecture.neurons
        
        # Optimized: filter hidden neurons once and cache neuron type info
        hidden_neurons = [nid for nid, neuron in neurons.items()
                         if neuron.neuron_type == NeuronType.HIDDEN]
        isolated_hidden = self.action_space._get_isolated_hidden_neurons(architecture)
        
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

        if phase == 0:  # EXPANDING
            if num_neurons < self.max_neurons:
                action_mask[ActionType.ADD_NEURON.value] = 0
        elif phase == 1:  # REFINEMENT
            if num_hidden > 0:
                action_mask[ActionType.MODIFY_ACTIVATION.value] = 0
            if num_neurons >= 2:
                action_mask[ActionType.ADD_CONNECTION.value] = 0
        elif phase == 2:  # PRUNING
            if len(isolated_hidden) > 0:
                action_mask[ActionType.REMOVE_NEURON.value] = 0
            if num_connections > 0:
                action_mask[ActionType.REMOVE_CONNECTION.value] = 0

        # Build masks (size = tensor_size). We return multiple masks for different action uses
        # Generic source mask: allowed for ADD_CONNECTION sampling (exclude OUTPUT neurons)
        source_mask = torch.full((tensor_size,), -1e9)
        valid_source_ids = [nid for nid, neuron in neurons.items() if nid < tensor_size and neuron.neuron_type != NeuronType.OUTPUT]
        if valid_source_ids:
            source_mask[valid_source_ids] = 0

        # Remove-specific source mask: only isolated hidden neurons can be removed
        remove_source_mask = torch.full((tensor_size,), -1e9)
        if isolated_hidden:
            valid_remove_ids = [nid for nid in isolated_hidden if nid < tensor_size]
            if valid_remove_ids:
                remove_source_mask[valid_remove_ids] = 0

        # Modify-specific source mask: any hidden neuron may be modified
        modify_source_mask = torch.full((tensor_size,), -1e9)
        valid_modify_ids = [nid for nid in hidden_neurons if nid < tensor_size]
        if valid_modify_ids:
            modify_source_mask[valid_modify_ids] = 0

        # Target neuron mask for ADD_CONNECTION: exclude INPUT neurons
        target_mask = torch.full((tensor_size,), -1e9)
        valid_target_ids = [nid for nid, neuron in neurons.items() if nid < tensor_size and neuron.neuron_type != NeuronType.INPUT]
        if valid_target_ids:
            target_mask[valid_target_ids] = 0

        return {
            'action_type': action_mask,
            'source_neurons': source_mask,
            'remove_source_neurons': remove_source_mask,
            'modify_source_neurons': modify_source_mask,
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
        """Select action using policy output with masking and conditional distributions"""

        masks_dict = self.get_action_masks(architecture, phase=0)  # phase can be adjusted as needed
        masks = masks_dict.copy()
        tensor_size = masks['tensor_size']
        
        # Move masks to the same device as policy_output
        device = policy_output['action_type'].device
        for key in masks:
            if torch.is_tensor(masks[key]):
                masks[key] = masks[key].to(device)

        # Sample action type
        action_logits = policy_output['action_type'] + masks['action_type']

        if exploration:
            action_type_idx = Categorical(logits=action_logits).sample().item()
        else:
            action_type_idx = action_logits.argmax().item()

        action_type = ActionType(action_type_idx)
        
        # Get conditional logits for this action type
        conditional_logits = self._compute_conditional_logits(policy_output, action_type)
        
        # Select parameters based on action type - optimized with early returns
        if action_type == ActionType.ADD_NEURON:
            activation_logits = conditional_logits['activation_logits']
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
                conditional_logits['source_logits'], tensor_size, masks.get('remove_source_neurons', masks['source_neurons'])
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
                conditional_logits['source_logits'], tensor_size, masks.get('modify_source_neurons', masks['source_neurons'])
            )
            activation_logits = conditional_logits['activation_logits']
            
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
            # Get conditional logits with no source initially
            conditional_logits_no_source = self._compute_conditional_logits(policy_output, action_type)
            source_logits = self._prepare_logits(
                conditional_logits_no_source['source_logits'], tensor_size, masks['source_neurons']
            )
            
            # Sample source first
            if exploration:
                source_idx = Categorical(logits=source_logits).sample().item()
            else:
                source_idx = source_logits.argmax().item()

            # Now get target logits conditioned on the chosen source
            conditional_logits_with_source = self._compute_conditional_logits(policy_output, action_type, source_idx)
            target_logits = self._prepare_logits(
                conditional_logits_with_source['target_logits'], tensor_size, masks['target_neurons']
            )
            
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

        # Get conditional source logits
        conditional_logits_no_source = self._compute_conditional_logits(policy_output, action_type)
        source_logits_full = conditional_logits_no_source['source_logits']
        
        # For remove_connection, we need to evaluate each existing connection
        device = source_logits_full.device
        
        # Pad to tensor_size if needed - optimized
        if source_logits_full.shape[1] < tensor_size:
            source_logits_full = torch.cat([
                source_logits_full, 
                torch.full((source_logits_full.shape[0], tensor_size - source_logits_full.shape[1]), -1e9, device=device)
            ], dim=1)
        else:
            source_logits_full = source_logits_full[:, :tensor_size]

        # Vectorized scoring: build tensors of source/target IDs, then index in parallel
        source_ids = torch.tensor([conn.source_id for conn in existing_connections], device=device)
        target_ids = torch.tensor([conn.target_id for conn in existing_connections], device=device)
        
        # Clamp indices to valid range to prevent indexing errors
        source_ids = source_ids.clamp(0, tensor_size - 1)
        target_ids = target_ids.clamp(0, tensor_size - 1)
        
        # Get source scores
        source_scores = source_logits_full[0, source_ids]  # Assuming batch size 1
        
        # Get target scores conditioned on each source
        combined_logits = torch.zeros(len(existing_connections), device=device)
        for i, (source_id, target_id) in enumerate(zip(source_ids, target_ids)):
            conditional_logits_with_source = self._compute_conditional_logits(policy_output, action_type, source_id.item())
            target_logits_full = conditional_logits_with_source['target_logits']
            
            # Pad target logits if needed
            if target_logits_full.shape[1] < tensor_size:
                target_logits_full = torch.cat([
                    target_logits_full,
                    torch.full((target_logits_full.shape[0], tensor_size - target_logits_full.shape[1]), -1e9, device=device)
                ], dim=1)
            else:
                target_logits_full = target_logits_full[:, :tensor_size]
            
            target_score = target_logits_full[0, target_id]
            combined_logits[i] = source_scores[i] + target_score

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
