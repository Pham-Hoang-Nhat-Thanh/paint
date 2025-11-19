import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_transformer import GraphTransformer
import time
from typing import Dict, List
from blueprint_modules.network import NeuralArchitecture, NeuronType, ActivationType
from blueprint_modules.action import ActionType, Action, ActionSpace

# Enable TF32 for faster matrix multiplications on Ampere+ GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class UnifiedPolicyValueNetwork(nn.Module):
    """A network that predicts both policy and value from a graph representation.

    This network uses a GraphTransformer to encode the neural architecture and
    outputs a policy (logits for different actions) and a value (a scalar
    prediction of the architecture's quality).

    Attributes:
        hidden_dim (int): The dimensionality of hidden layers.
        max_neurons (int): The maximum number of neurons the policy heads can
            handle.
        num_actions (int): The number of discrete action types.
        num_activations (int): The number of activation function types.
    """
    
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
        """Performs the forward pass for the policy-value network.

        Args:
            graph_data (Dict): A dictionary containing the graph representation
                of the neural architecture.
            phase (int): The current evolutionary phase, used for conditioning
                the network.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the policy logits
            for various action components and the predicted value.
        """
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
    """Handles action selection, masking, and validation.

    This class is responsible for generating action masks to prevent invalid
    actions, computing action probabilities based on the policy network's
    output, and providing a clean interface for action selection during MCTS.

    Attributes:
        max_neurons (int): The maximum number of neurons supported.
        action_space (ActionSpace): The action space definition.
        exploration_boost (float): A factor to boost exploration of
            under-represented actions.
    """

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
        """Compute priors using correct conditional probabilities in a fully vectorized form."""
        if not actions:
            return torch.tensor([], device=next(iter(policy_output.values())).device)

        device = policy_output['action_type'].device
        n_actions = len(actions)
        tensor_size = int(masks['tensor_size'])

        step1 = time.time()
        # --- 1. Initial Setup & Action Type Log-Prob ---
        action_type_logits = policy_output['action_type'].squeeze(0)
        action_type_mask = masks['action_type'].to(device)
        log_at = F.log_softmax(action_type_logits + action_type_mask, dim=-1)

        at_vals = torch.tensor([a.action_type.value for a in actions], device=device, dtype=torch.long)
        logp = log_at[at_vals]
        step1 = time.time() - step1

        step2 = time.time()
        # --- 2. Pre-extract all action metadata ---
        src_vals = torch.tensor([a.source_neuron if a.source_neuron is not None else -1 for a in actions], device=device, dtype=torch.long)
        tgt_vals = torch.tensor([a.target_neuron if a.target_neuron is not None else -1 for a in actions], device=device, dtype=torch.long)
        
        activation_types = list(ActivationType)
        act_vals = torch.tensor([activation_types.index(a.activation) if a.activation is not None else -1 for a in actions], device=device, dtype=torch.long)
        step2 = time.time() - step2

        step3 = time.time()
        # --- 3. Batch process each action type ---
        for at_enum in ActionType:
            at_val = at_enum.value
            action_mask = (at_vals == at_val)
            if not action_mask.any():
                continue

            at_name = self._get_action_type_name(at_val)
            at_src_vals = src_vals[action_mask]
            
            step3a = time.time()
            # --- Source Logits ---
            if at_src_vals.ge(0).any():
                src_logits = policy_output['source_logits_dict'].get(at_name)
                if src_logits is not None:
                    # Select appropriate mask
                    if at_enum == ActionType.REMOVE_NEURON:
                        src_mask_tensor = masks.get('remove_source_neurons', masks['source_neurons']).to(device)
                    elif at_enum == ActionType.MODIFY_ACTIVATION:
                        src_mask_tensor = masks.get('modify_source_neurons', masks['source_neurons']).to(device)
                    else:
                        src_mask_tensor = masks['source_neurons'].to(device)

                    src_logits_prepped = self._prepare_logits(src_logits, tensor_size, src_mask_tensor).squeeze(0)
                    log_src_probs = F.log_softmax(src_logits_prepped, dim=-1)
                    
                    valid_src_indices = at_src_vals.clamp(0, tensor_size - 1)
                    logp[action_mask] += log_src_probs[valid_src_indices]
            step3a = time.time() - step3a

            step3b = time.time()
            # --- Target & Activation Logits (conditionally) ---
            needs_target = 'target_heads' in policy_output and at_name in policy_output['target_heads']
            needs_activation = 'activation_heads' in policy_output and at_name in policy_output['activation_heads']

            if needs_target or (needs_activation and at_name == 'modify_activation'):
                # Get unique source neurons for this action type
                unique_at_srcs, inv_idx = torch.unique(at_src_vals[at_src_vals >= 0], return_inverse=True)
                
                if unique_at_srcs.numel() > 0:
                    # Batch-compute conditional logits
                    batched_logits = self._compute_batched_conditional_logits(
                        policy_output, at_enum, unique_at_srcs
                    )
                    # Distribute target log-probs
                    if needs_target and 'target_logits' in batched_logits:
                        target_mask_full = masks['target_neurons'].to(device)
                        tgt_logits_prepped = self._prepare_logits(batched_logits['target_logits'], tensor_size, target_mask_full)
                        log_tgt_probs = F.log_softmax(tgt_logits_prepped, dim=-1)
                        
                        at_tgt_vals = tgt_vals[action_mask]
                        valid_tgt_mask = at_tgt_vals >= 0
                        if valid_tgt_mask.any():
                           tgt_indices = at_tgt_vals[valid_tgt_mask].clamp(0, tensor_size - 1)
                           # Select correct log_prob row for each action via inv_idx
                           logp_slice = torch.zeros_like(at_tgt_vals, dtype=torch.float)
                           logp_slice[valid_tgt_mask] = log_tgt_probs[inv_idx, tgt_indices]
                           logp[action_mask] += logp_slice

                    # Distribute activation log-probs (for modify_activation)
                    if needs_activation and 'activation_logits' in batched_logits:
                        act_mask_full = (masks.get('activation', torch.zeros(4, device=device)).to(device) == 0).float() * -1e9
                        act_logits_prepped = batched_logits['activation_logits'] + act_mask_full
                        log_act_probs = F.log_softmax(act_logits_prepped, dim=-1)
                        
                        at_act_vals = act_vals[action_mask]
                        valid_act_mask = at_act_vals >= 0
                        if valid_act_mask.any():
                           act_indices = at_act_vals[valid_act_mask]
                           logp_slice = torch.zeros_like(at_act_vals, dtype=torch.float)
                           logp_slice[valid_act_mask] = log_act_probs[inv_idx, act_indices]
                           logp[action_mask] += logp_slice

            elif needs_activation: # For 'add_neuron'
                act_logits = policy_output['activation_heads'][at_name](policy_output['shared_features']).squeeze(0)
                act_mask_full = (masks.get('activation', torch.zeros(act_logits.shape[-1], device=device)).to(device) == 0).float() * -1e9
                log_act_probs = F.log_softmax(act_logits + act_mask_full, dim=-1)

                at_act_vals = act_vals[action_mask]
                valid_act_mask = at_act_vals >= 0
                if valid_act_mask.any():
                    act_indices = at_act_vals[valid_act_mask]
                    # Use the safer slice assignment to avoid size mismatch errors
                    logp_slice = torch.zeros_like(at_act_vals, dtype=torch.float)
                    logp_slice[valid_act_mask] = log_act_probs[act_indices]
                    logp[action_mask] += logp_slice
            step3b = time.time() - step3b
        step3 = time.time() - step3

        #print("---- Action Prior Computation Timing ----")
        #print(f"Step times: Step1={step1:.4f}s, Step2={step2:.4f}s, Step3={step3:.4f}s")
        #print(f"  Step3 breakdown: Source={step3a:.4f}s, Target/Activation={step3b:.4f}s")
        #print("----\n")

        return torch.exp(logp)

    def _compute_batched_conditional_logits(self, policy_output: Dict, action_type: ActionType, 
                                            source_neurons: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute conditional logits for a batch of source neurons."""
        shared_features = policy_output['shared_features']
        action_type_name = self._get_action_type_name(action_type.value)
        result = {}
        
        # Batch encode source neurons
        source_one_hot = F.one_hot(source_neurons, num_classes=self.max_neurons).float()
        source_features = policy_output['source_encoder'](source_one_hot)

        # Expand shared features to match batch size
        num_sources = source_features.shape[0]
        expanded_shared = shared_features.expand(num_sources, -1)
        
        conditioned_features = torch.cat([expanded_shared, source_features], dim=-1)

        # Compute target logits in batch
        if 'target_heads' in policy_output and action_type_name in policy_output['target_heads']:
            result['target_logits'] = policy_output['target_heads'][action_type_name](conditioned_features)
            
        # Compute activation logits in batch (for modify_activation)
        if 'activation_heads' in policy_output and action_type_name == 'modify_activation':
            result['activation_logits'] = policy_output['activation_heads'][action_type_name](conditioned_features)
            
        return result

    def get_action_masks(self, architecture: NeuralArchitecture, phase: int) -> Dict[str, torch.Tensor]:
        """Generates masks to prevent illegal actions based on the architecture.

        This method creates several boolean masks that indicate which actions
        are valid for the current architectural state and evolutionary phase.

        Args:
            architecture (NeuralArchitecture): The current neural architecture.
            phase (int): The current evolutionary phase.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of masks for different action
            components.
        """
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
