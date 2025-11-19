import numpy as np
from typing import List, Dict, Any, Tuple
import torch
import copy

class ExperienceReplay:
    """Implements a prioritized experience replay buffer.

    This buffer stores experiences (state, action, reward, etc.) and allows
    for sampling of experiences with prioritization, which can lead to more
    efficient training.

    Attributes:
        capacity (int): The maximum number of experiences to store.
        alpha (float): The prioritization exponent.
        beta (float): The importance sampling exponent.
        buffer (List): The list of stored experiences.
        priorities (np.ndarray): The priorities of the experiences.
        position (int): The current position in the buffer.
        size (int): The number of experiences in the buffer.
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
    def add(self, experience: Dict[str, Any], priority: float = None):
        """Adds an experience to the replay buffer.

        Args:
            experience (Dict[str, Any]): The experience to add.
            priority (float, optional): The priority of the experience. If
                None, the maximum existing priority is used.
        """
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
        """Samples a batch of experiences from the buffer.

        The sampling is done according to the priorities of the experiences.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            Tuple[List[Dict], List[int], torch.Tensor]: A tuple containing the
            sampled experiences, their indices, and the importance sampling
            weights.
        """
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
        """Updates the priorities of a batch of experiences.

        Args:
            indices (List[int]): The indices of the experiences to update.
            priorities (List[float]): The new priorities for the experiences.
        """
        for idx, priority in zip(indices, priorities):
            if idx < self.size:
                self.priorities[idx] = priority + 1e-6  # Small epsilon to avoid zero priority
    
    def __len__(self):
        return self.size
    
    def is_ready(self, min_size: int) -> bool:
        """Checks if the buffer has enough experiences to start training.

        Args:
            min_size (int): The minimum number of experiences required.

        Returns:
            bool: True if the buffer is ready, False otherwise.
        """
        return self.size >= min_size

    def state_dict(self):
        """Returns the state of the replay buffer for checkpointing.

        Returns:
            dict: A dictionary containing the state of the buffer.
        """
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
        """Restores the state of the replay buffer from a checkpoint.

        Args:
            state (dict): The state dictionary to load.
        """
        self.capacity = state.get('capacity', 10000)
        self.alpha = state.get('alpha', 0.6)
        self.beta = state.get('beta', 0.4)
        self.buffer = state.get('buffer', [])
        self.priorities = np.array(state.get('priorities', [1.0]*self.capacity), dtype=np.float32)
        self.position = state.get('position', 0)
        self.size = state.get('size', len(self.buffer))

    def clear(self):
        """Removes all experiences from the buffer."""
        self.buffer = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0