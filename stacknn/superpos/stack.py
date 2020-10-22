from overrides import overrides
import torch
from typing import Optional

from .base import AbstractStack
from . import functional as F


class Stack(AbstractStack):

    """Implements a superposition-y differentiable stack architecture inspired by Suzgun et al.,
    2019. The paper link is:
    https://arxiv.org/abs/1911.03329v1

    Unlike the model in stack.py, this model enforces that the stack must push or pop at each time
    step. In other words, it does not allow no-operation as an option.
    """

    @overrides
    def update(self,
               policies: torch.FloatTensor,  # Distribution of shape [batch_size, 2].
               new_vecs: torch.FloatTensor   # Vectors of shape [batch_size, stack_dim].
              ) -> torch.FloatTensor:
        self.tapes = F.update_stack(self.tapes, policies, new_vecs, self.max_depth)
        return self.tapes

    @classmethod
    @overrides
    def get_num_actions(cls) -> int:
        return 2
