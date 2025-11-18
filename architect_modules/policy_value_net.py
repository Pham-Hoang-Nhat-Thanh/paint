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
            if x is None:
                return None
            if torch.is_tensor(x) and x.dim() > 1 and x.shape[0] == 1:
                return x.squeeze(0)
            return x

        # Compute log-prob of action types once
        action_type_logits = _maybe_squeeze(policy_output['action_type'])
        action_type_mask = masks['action_type'].to(device)
        log_at = F.log_softmax(action_type_logits + action_type_mask, dim=-1)  # [C]

        # Build flat tensors for action metadata
        at_vals = torch.tensor([a.action_type.value for a in actions], dtype=torch.long, device=device)
        src_vals = torch.tensor([(-1 if a.source_neuron is None else int(a.source_neuron)) for a in actions], dtype=torch.long, device=device)
        tgt_vals = torch.tensor([(-1 if a.target_neuron is None else int(a.target_neuron)) for a in actions], dtype=torch.long, device=device)
        act_vals = torch.tensor([(-1 if a.activation is None else int(list(ActivationType).index(a.activation))) for a in actions], dtype=torch.long, device=device)

        # Base log-prob from action type
        logp = log_at[at_vals]  # [n_actions]

        # Group unique (action_type, source) pairs to compute conditional logits once
        keys = torch.stack([at_vals, src_vals], dim=1)  # [n,2]
        unique_keys, inv_idx = torch.unique(keys, dim=0, return_inverse=True)

        # Prepare containers for log-prob contributions per unique group
        G = unique_keys.shape[0]
        logp_src_per_group = torch.zeros(G, device=device)

        # Precompute masks to use when preparing logits
        tensor_size = int(masks['tensor_size'])
        source_mask_full = masks['source_neurons'].to(device)
        remove_source_mask_full = masks.get('remove_source_neurons', masks['source_neurons']).to(device)
        modify_source_mask_full = masks.get('modify_source_neurons', masks['source_neurons']).to(device)
        target_mask_full = masks['target_neurons'].to(device)
        # Determine activation mask: prefer provided masks, otherwise try to infer
        activation_mask_full = masks.get('activation', None)
        if activation_mask_full is None:
            # Try to infer activation size from policy_output if available
            inferred_len = None
            ao = policy_output.get('activation_logits', None)
            if torch.is_tensor(ao):
                inferred_len = ao.shape[-1]
            else:
                # Fallback to previously assumed default of 4 activations
                inferred_len = 4
            activation_mask_full = torch.zeros(inferred_len, device=device)
        else:
            activation_mask_full = activation_mask_full.to(device)

        # Cache for conditional logits per unique group key. First check expand-step cache
        cond_logits_cache = {}

        # Precompute list of unique source indices (for precomputing source embeddings)
        unique_srcs = unique_keys[:, 1]
        unique_srcs_pos = unique_srcs[unique_srcs >= 0]
        if unique_srcs_pos.numel() > 0 and 'source_encoder' in policy_output:
            unique_src_vals = torch.unique(unique_srcs_pos)
            # Precompute one-hot encodings and source features in batch
            one_hot = F.one_hot(unique_src_vals.to(device), num_classes=self.max_neurons).float()
            precomp_feats = policy_output['source_encoder'](one_hot)
            # Map from source id -> precomputed feature (unsqueezed for concat compatibility)
            src_to_feat = {int(s.item()): precomp_feats[i].unsqueeze(0) for i, s in enumerate(unique_src_vals)}
        else:
            src_to_feat = {}

        # Iterate groups and compute/cached conditional logits (use expand cache if available)
        log_target_rows = torch.full((G, tensor_size), -1e9, device=device)
        log_activation_rows = torch.full((G, activation_mask_full.shape[0] if activation_mask_full is not None else self.num_activations), -1e9, device=device)

        for g_idx in range(G):
            a_val = int(unique_keys[g_idx, 0].item())
            s_val = int(unique_keys[g_idx, 1].item())
            action_type_enum = ActionType(a_val)

            # Decide appropriate source mask for this action type
            if action_type_enum == ActionType.REMOVE_NEURON:
                src_mask = remove_source_mask_full
            elif action_type_enum == ActionType.MODIFY_ACTIVATION:
                src_mask = modify_source_mask_full
            else:
                src_mask = source_mask_full

            key = (a_val, s_val)
            # First check global expand-step cache (across sibling expansions)
            if key in self._expand_step_cache:
                cached = self._expand_step_cache[key]
            elif key in cond_logits_cache:
                cached = cond_logits_cache[key]
            else:
                # Use precomputed source feature when available to avoid one-hot creation
                precomputed_feat = None
                if s_val >= 0:
                    precomputed_feat = src_to_feat.get(s_val, None)
                cond = self._compute_conditional_logits(policy_output, action_type_enum, None if s_val < 0 else s_val,
                                                       precomputed_source_features=precomputed_feat)
                # Squeeze singleton batch dims for easier indexing
                for k, v in list(cond.items()):
                    if torch.is_tensor(v):
                        cond[k] = _maybe_squeeze(v)

                # Build cached entry with masked log-probs
                cached = {}
                # Source log-prob handled below
                if 'target_logits' in cond and cond['target_logits'] is not None:
                    tgt_logits = cond['target_logits']
                    tgt_logits_prepped = self._prepare_logits(tgt_logits.unsqueeze(0), tensor_size, target_mask_full).squeeze(0)
                    cached['_log_target'] = F.log_softmax(tgt_logits_prepped, dim=-1)
                if 'activation_logits' in cond and cond['activation_logits'] is not None:
                    act_logits = cond['activation_logits']
                    act_logits_prepped = (act_logits + (activation_mask_full == 0).float() * -1e9) if act_logits.dim() == 1 else act_logits
                    cached['_log_activation'] = F.log_softmax(act_logits_prepped, dim=-1)

                # Store raw source logits (if present) so we can compute log probability for the chosen source
                if 'source_logits' in cond and cond['source_logits'] is not None:
                    cached['_raw_source_logits'] = cond['source_logits']

                # Put into both local and expand cache for reuse
                cond_logits_cache[key] = cached
                self._expand_step_cache[key] = cached

            # Compute source log-prob for this group if applicable
            if s_val >= 0 and '_raw_source_logits' in cached:
                src_logits = cached['_raw_source_logits']
                src_logits_prepped = self._prepare_logits(src_logits.unsqueeze(0), tensor_size, src_mask).squeeze(0)
                log_src = F.log_softmax(src_logits_prepped, dim=-1)
                idx = min(s_val, tensor_size - 1)
                logp_src_per_group[g_idx] = log_src[idx]
            else:
                logp_src_per_group[g_idx] = 0.0

            # If cached target/activation present, copy into group rows (already masked)
            if '_log_target' in cached:
                log_target_rows[g_idx, :cached['_log_target'].shape[0]] = cached['_log_target'][:tensor_size]
            if '_log_activation' in cached:
                act_row = cached['_log_activation']
                log_activation_rows[g_idx, :act_row.shape[0]] = act_row

        # Map group source log-probs back to actions
        logp_src = logp_src_per_group[inv_idx]
        logp = logp + logp_src

        # Vectorized mapping for target and activation using stacked cached rows
        # Targets
        tgt_mask = tgt_vals >= 0
        if tgt_mask.any():
            tgt_indices = tgt_vals.clamp(min=0, max=tensor_size - 1)
            gathered_targets = log_target_rows[inv_idx, tgt_indices]
            # Zero out positions where target not applicable
            gathered_targets = gathered_targets * tgt_mask.to(device)
        else:
            gathered_targets = torch.zeros(n_actions, device=device)

        # Activations
        act_mask = act_vals >= 0
        if act_mask.any():
            act_indices = act_vals.clamp(min=0, max=log_activation_rows.shape[1] - 1)
            gathered_acts = log_activation_rows[inv_idx, act_indices]
            gathered_acts = gathered_acts * act_mask.to(device)
        else:
            gathered_acts = torch.zeros(n_actions, device=device)

        logp = logp + gathered_targets + gathered_acts

        # Convert log-probs to linear space priors
        priors = torch.exp(logp)
        # Normalize or floor if desired (keep as raw priors)
        return priors

    def _compute_conditional_logits(self, policy_output: Dict, action_type: ActionType, 
                                  source_neuron: int = None, precomputed_source_features: torch.Tensor = None) -> Dict[str, torch.Tensor]:
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
            # Use precomputed source features when provided to avoid one-hot creation
            if precomputed_source_features is not None:
                source_features = precomputed_source_features
                if torch.is_tensor(source_features) and source_features.dim() == 1:
                    source_features = source_features.unsqueeze(0)
            else:
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
                # Condition on source for modify_activation; use precomputed feature if available
                if precomputed_source_features is not None:
                    source_features = precomputed_source_features
                    if torch.is_tensor(source_features) and source_features.dim() == 1:
                        source_features = source_features.unsqueeze(0)
                else:
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
