from overrides import overrides
import torch
from typing import Optional

from .base import AbstractStack
from . import functional as F


class RewriteStack(AbstractStack):

    """This stack is similar to the NoOp stack, but NoOp is replaced by an operation that rewrites
    the top element of the stack.
    """

    @overrides
    def update(self,
               policies: torch.FloatTensor,  # Distribution of shape [batch_size, 3].
               new_vecs: torch.FloatTensor   # Vectors of shape [batch_size, stack_dim].
              ) -> torch.FloatTensor:
        self.tapes = F.update_rewrite_stack(self.tapes, policies, new_vecs, self.max_depth)
        return self.tapes

    @classmethod
    @overrides
    def get_num_actions(cls) -> int:
        return 3
