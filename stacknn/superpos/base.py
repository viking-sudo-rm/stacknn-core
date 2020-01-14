from abc import ABCMeta, abstractmethod
import torch
from typing import Optional


class AbstractStack(metaclass=ABCMeta):

    def __init__(self, stack_dim: int):
        self.stack_dim = stack_dim
        self.tapes: torch.FloatTensor = None

    @classmethod
    def empty(cls, batch_size: int, stack_dim: int, device: Optional[int] = None):
        stack = cls(stack_dim)
        stack.reset(batch_size, device)
        return stack

    def reset(self, batch_size: int, device: Optional[int] = None):
        self.tapes = torch.zeros(batch_size, 0, self.stack_dim, device=device)
        self.device = device

    @abstractmethod
    def update(self,
               policies: torch.Tensor,  # Distribution of shape [batch_size, num_actions].
               new_vecs: torch.Tensor   # Vectors of shape [batch_size, stack_dim].
              ):
        return NotImplemented

    @abstractmethod
    def get_num_actions(self):
        """This can be either a class or instance method depending on the stack type."""
        return NotImplemented
