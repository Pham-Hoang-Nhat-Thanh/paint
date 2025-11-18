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
        # Cache used to share computed conditional logits across calls
        # within an MCTS expand step to avoid recomputation for sibling nodes.
        # Cleared by calling `start_expand_step()` or `end_expand_step()`.
        self._expand_step_cache = {}

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

        # Helper: safely squeeze leading batch dim if present
        def _maybe_squeeze(x: torch.Tensor) -> torch.Tensor:
            return x.squeeze(0) if x is not None and x.dim() > 1 and x.shape[0] == 1 else x

        # Compute log-prob of action types once
        action_type_logits = _maybe_squeeze(policy_output['action_type'])
        action_type_mask = masks['action_type'].to(device)
        log_at = F.log_softmax(action_type_logits + action_type_mask, dim=-1)

        # Build flat tensors for action metadata (single pass extraction)
        at_vals = torch.empty(n_actions, dtype=torch.long, device=device)
        src_vals = torch.empty(n_actions, dtype=torch.long, device=device)
        tgt_vals = torch.empty(n_actions, dtype=torch.long, device=device)
        act_vals = torch.empty(n_actions, dtype=torch.long, device=device)
        
        activation_types = list(ActivationType)
        for i, a in enumerate(actions):
            at_vals[i] = a.action_type.value
            src_vals[i] = -1 if a.source_neuron is None else a.source_neuron
            tgt_vals[i] = -1 if a.target_neuron is None else a.target_neuron
            act_vals[i] = -1 if a.activation is None else activation_types.index(a.activation)

        # Base log-prob from action type
        logp = log_at[at_vals]

        # Group unique (action_type, source) pairs
        keys = torch.stack([at_vals, src_vals], dim=1)
        unique_keys, inv_idx = torch.unique(keys, dim=0, return_inverse=True)
        G = unique_keys.shape[0]

        # Prepare masks and constants once
        tensor_size = int(masks['tensor_size'])
        source_mask_full = masks['source_neurons'].to(device)
        remove_source_mask_full = masks.get('remove_source_neurons', source_mask_full).to(device)
        modify_source_mask_full = masks.get('modify_source_neurons', source_mask_full).to(device)
        target_mask_full = masks['target_neurons'].to(device)
        
        activation_mask_full = masks.get('activation')
        if activation_mask_full is None:
            ao = policy_output.get('activation_logits')
            inferred_len = ao.shape[-1] if torch.is_tensor(ao) else 4
            activation_mask_full = torch.zeros(inferred_len, device=device)
        else:
            activation_mask_full = activation_mask_full.to(device)
        
        act_mask_full_neg = (activation_mask_full == 0).float() * -1e9

        # Precompute source embeddings in batch
        unique_srcs = unique_keys[:, 1]
        unique_srcs_pos = unique_srcs[unique_srcs >= 0]
        src_to_feat = {}
        
        if unique_srcs_pos.numel() > 0 and 'source_encoder' in policy_output:
            unique_src_vals = torch.unique(unique_srcs_pos)
            one_hot = F.one_hot(unique_src_vals, num_classes=self.max_neurons).float()
            precomp_feats = policy_output['source_encoder'](one_hot)
            src_to_feat = {int(s.item()): precomp_feats[i:i+1] for i, s in enumerate(unique_src_vals)}

        # Process groups
        logp_src_per_group = torch.zeros(G, device=device)
        log_target_rows = torch.full((G, tensor_size), -1e9, device=device)
        log_activation_rows = torch.full((G, activation_mask_full.shape[0]), -1e9, device=device)

        for g_idx in range(G):
            a_val = int(unique_keys[g_idx, 0].item())
            s_val = int(unique_keys[g_idx, 1].item())
            action_type_enum = ActionType(a_val)

            # Select appropriate mask
            if action_type_enum == ActionType.REMOVE_NEURON:
                src_mask = remove_source_mask_full
            elif action_type_enum == ActionType.MODIFY_ACTIVATION:
                src_mask = modify_source_mask_full
            else:
                src_mask = source_mask_full

            key = (a_val, s_val)
            
            # Check cache
            if key in self._expand_step_cache:
                cached = self._expand_step_cache[key]
            else:
                precomputed_feat = src_to_feat.get(s_val) if s_val >= 0 else None
                cond = self._compute_conditional_logits(policy_output, action_type_enum, 
                                                    None if s_val < 0 else s_val,
                                                    precomputed_source_features=precomputed_feat)
                
                # Squeeze and cache
                cached = {}
                for k, v in cond.items():
                    if torch.is_tensor(v):
                        cond[k] = _maybe_squeeze(v)

                if 'target_logits' in cond and cond['target_logits'] is not None:
                    tgt_logits_prepped = self._prepare_logits(cond['target_logits'].unsqueeze(0), 
                                                            tensor_size, target_mask_full).squeeze(0)
                    cached['_log_target'] = F.log_softmax(tgt_logits_prepped, dim=-1)
                
                if 'activation_logits' in cond and cond['activation_logits'] is not None:
                    act_logits = cond['activation_logits']
                    act_logits_prepped = act_logits + act_mask_full_neg if act_logits.dim() == 1 else act_logits
                    cached['_log_activation'] = F.log_softmax(act_logits_prepped, dim=-1)
                
                if 'source_logits' in cond and cond['source_logits'] is not None:
                    cached['_raw_source_logits'] = cond['source_logits']

                self._expand_step_cache[key] = cached

            # Compute source log-prob
            if s_val >= 0 and '_raw_source_logits' in cached:
                src_logits_prepped = self._prepare_logits(cached['_raw_source_logits'].unsqueeze(0), 
                                                        tensor_size, src_mask).squeeze(0)
                log_src = F.log_softmax(src_logits_prepped, dim=-1)
                logp_src_per_group[g_idx] = log_src[min(s_val, tensor_size - 1)]

            # Copy cached rows
            if '_log_target' in cached:
                log_target_rows[g_idx, :cached['_log_target'].shape[0]] = cached['_log_target'][:tensor_size]
            if '_log_activation' in cached:
                act_row = cached['_log_activation']
                log_activation_rows[g_idx, :act_row.shape[0]] = act_row

        # Vectorized gather operations
        logp += logp_src_per_group[inv_idx]
        
        tgt_mask = tgt_vals >= 0
        if tgt_mask.any():
            tgt_indices = tgt_vals.clamp(0, tensor_size - 1)
            gathered_targets = log_target_rows[inv_idx, tgt_indices] * tgt_mask.float()
            logp += gathered_targets
        
        act_mask = act_vals >= 0
        if act_mask.any():
            act_indices = act_vals.clamp(0, log_activation_rows.shape[1] - 1)
            gathered_acts = log_activation_rows[inv_idx, act_indices] * act_mask.float()
            logp += gathered_acts

        return torch.exp(logp)


    def _compute_conditional_logits(self, policy_output: Dict, action_type: ActionType, 
                                source_neuron: int = None, precomputed_source_features: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Compute conditional logits given action type and optional source"""
        shared_features = policy_output['shared_features']
        action_type_name = self._get_action_type_name(action_type.value)
        result = {}
        
        # Source logits
        if 'source_logits_dict' in policy_output and action_type_name in policy_output['source_logits_dict']:
            result['source_logits'] = policy_output['source_logits_dict'][action_type_name]
        else:
            result['source_logits'] = policy_output.get('source_logits', 
                torch.zeros(shared_features.shape[0], self.max_neurons, device=shared_features.device))
        
        # Target logits
        if source_neuron is not None and 'target_heads' in policy_output and action_type_name in policy_output['target_heads']:
            if precomputed_source_features is not None:
                source_features = precomputed_source_features if precomputed_source_features.dim() > 1 else precomputed_source_features.unsqueeze(0)
            else:
                source_one_hot = F.one_hot(torch.tensor([source_neuron], device=shared_features.device), 
                                        self.max_neurons).float()
                source_features = policy_output['source_encoder'](source_one_hot)
            
            conditioned_features = torch.cat([shared_features, source_features], dim=-1)
            result['target_logits'] = policy_output['target_heads'][action_type_name](conditioned_features)
        else:
            result['target_logits'] = policy_output.get('target_logits', 
                torch.zeros(shared_features.shape[0], self.max_neurons, device=shared_features.device))
        
        # Activation logits
        if 'activation_heads' in policy_output and action_type_name in policy_output['activation_heads']:
            if action_type_name == 'modify_activation' and source_neuron is not None:
                if precomputed_source_features is not None:
                    source_features = precomputed_source_features if precomputed_source_features.dim() > 1 else precomputed_source_features.unsqueeze(0)
                else:
                    source_one_hot = F.one_hot(torch.tensor([source_neuron], device=shared_features.device), 
                                            self.max_neurons).float()
                    source_features = policy_output['source_encoder'](source_one_hot)
                conditioned_features = torch.cat([shared_features, source_features], dim=-1)
                result['activation_logits'] = policy_output['activation_heads'][action_type_name](conditioned_features)
            else:
                result['activation_logits'] = policy_output['activation_heads'][action_type_name](shared_features)
        else:
            result['activation_logits'] = policy_output.get('activation_logits', 
                torch.zeros(shared_features.shape[0], 4, device=shared_features.device))
        
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
