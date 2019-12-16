from abc import ABCMeta, abstractmethod
import torch
from typing import Optional


class AbstractStack(metaclass=ABCMeta):

    def __init__(self, tapes: torch.Tensor):
        self.tapes = tapes
        self.device = tapes.device

    @classmethod
    def empty(cls, batch_size: int, stack_dim: int, device: Optional[int] = None):
        tapes = torch.zeros(batch_size, 0, stack_dim, device=device)
        return cls(tapes)

    @abstractmethod
    def update(self,
               policies: torch.Tensor,  # Distribution of shape [batch_size, num_actions].
               new_vecs: torch.Tensor   # Vectors of shape [batch_size, stack_dim].
              ):
        return NotImplemented

    @classmethod
    @abstractmethod
    def get_num_actions(cls):
        # Use a method instead of a property for consistency with AllenNLP API.
        return NotImplemented
