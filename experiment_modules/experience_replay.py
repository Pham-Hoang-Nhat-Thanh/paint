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

class CurriculumExperienceReplay(ExperienceReplay):
    """Experience replay that handles curriculum learning stages"""
    
    def __init__(self, capacity: int = 10000):
        super().__init__(capacity)
        self.stage_experiences = {
            "supervised": [],
            "mixed": [],
            "self_play": []
        }
        
    def add_stage_experience(self, experience: Dict, stage: str, priority: float = None):
        """Add experience with stage information"""
        experience['stage'] = stage
        self.add(experience, priority)
        self.stage_experiences[stage].append(experience)
        
    def sample_curriculum_batch(self, batch_size: int, stage_weights: Dict[str, float] = None):
        """Sample batch with curriculum weighting across stages"""
        if stage_weights is None:
            stage_weights = {"supervised": 0.2, "mixed": 0.3, "self_play": 0.5}
            
        # Sample from each stage according to weights
        batch_experiences = []
        batch_indices = []
        
        for stage, weight in stage_weights.items():
            stage_size = max(1, int(batch_size * weight))
            if len(self.stage_experiences[stage]) >= stage_size:
                stage_samples = random.sample(self.stage_experiences[stage], stage_size)
                batch_experiences.extend(stage_samples)
                
        # If we don't have enough samples, fill with random ones
        while len(batch_experiences) < batch_size and self.size > 0:
            additional = self.sample(min(batch_size - len(batch_experiences), self.size))
            batch_experiences.extend(additional[0])
            
        return batch_experiences, [], torch.ones(len(batch_experiences))