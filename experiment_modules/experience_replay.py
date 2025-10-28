import numpy as np
import random
from typing import List, Dict, Any, Tuple
import torch

import numpy as np
import random
from typing import List, Dict, Any, Tuple
import torch

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
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
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
