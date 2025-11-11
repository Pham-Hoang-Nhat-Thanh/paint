import numpy as np
from typing import List, Dict, Any, Tuple
import torch
import copy

class ExperienceReplay:
    """Stores and samples training experiences with prioritization"""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
    def add(self, experience: Dict[str, Any], priority: float = None):
        """Add experience with priority"""
        if priority is None:
            # Default priority for new experiences
            priority = max(self.priorities.max(), 1.0) if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.priorities[self.position] = priority
            self.size += 1
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
            
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Dict], List[int], torch.Tensor]:
        """Sample batch of experiences with priority weighting"""
        if self.size == 0:
            return [], [], torch.tensor([])
            
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        # Get experiences (return deep copies to avoid accidental in-place mutation
        # of items stored in the buffer, which can introduce device inconsistencies)
        experiences = [copy.deepcopy(self.buffer[i]) for i in indices]
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** -self.beta
        weights = weights / weights.max()
        weights = torch.FloatTensor(weights)
        
        return experiences, indices.tolist(), weights
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            if idx < self.size:
                self.priorities[idx] = priority + 1e-6  # Small epsilon to avoid zero priority
    
    def __len__(self):
        return self.size
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training"""
        return self.size >= min_size

    def state_dict(self):
        """Serialize buffer and priorities for checkpointing"""
        return {
            'capacity': self.capacity,
            'alpha': self.alpha,
            'beta': self.beta,
            'buffer': self.buffer,
            'priorities': self.priorities.tolist(),
            'position': self.position,
            'size': self.size
        }

    def load_state_dict(self, state):
        """Restore buffer and priorities from checkpoint"""
        self.capacity = state.get('capacity', 10000)
        self.alpha = state.get('alpha', 0.6)
        self.beta = state.get('beta', 0.4)
        self.buffer = state.get('buffer', [])
        self.priorities = np.array(state.get('priorities', [1.0]*self.capacity), dtype=np.float32)
        self.position = state.get('position', 0)
        self.size = state.get('size', len(self.buffer))

    def clear(self):
        """Clear all experiences from buffer"""
        self.buffer = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0