from abc import ABCMeta, abstractmethod
import torch
from typing import Optional


class AbstractStack(metaclass=ABCMeta):

    def __init__(self, stack_dim: int, max_depth: Optional[int] = None):
        self.stack_dim = stack_dim
        self.max_depth = max_depth
        self.tapes: torch.FloatTensor = None

    @classmethod
    def empty(cls, batch_size: int, stack_dim: int, device: Optional[int] = None):
        stack = cls(stack_dim)
        stack.reset(batch_size, device)
        return stack

    def reset(self, batch_size: int, device: Optional[int] = None) -> None:
        del self.tapes
        self.tapes = torch.zeros(batch_size, 0, self.stack_dim, device=device)
        self.device = device

    def _enforce_max_depth(self) -> None:
        if self.max_depth is not None:
            self.tapes = self.tapes[:, :self.max_depth, :]

    @abstractmethod
    def update(self,
               policies: torch.FloatTensor,  # Distribution of shape [batch_size, num_actions].
               new_vecs: torch.FloatTensor   # Vectors of shape [batch_size, stack_dim].
              ) -> torch.FloatTensor:
        return NotImplemented

    @abstractmethod
    def get_num_actions(self):
        """This can be either a class or instance method depending on the stack type."""
        return NotImplemented
