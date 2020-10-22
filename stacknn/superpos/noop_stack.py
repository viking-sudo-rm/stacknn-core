from overrides import overrides
import torch
from typing import Optional

from .base import AbstractStack
from . import functional as F


class NoOpStack(AbstractStack):

    """Implements a superposition-y differentiable stack architecture inspired by Suzgun et al.,
    2019. The paper link is:
    https://arxiv.org/abs/1911.03329v1

    This stack is extended to allow no operation.
    """

    @overrides
    def update(self,
               policies: torch.FloatTensor,  # Distribution of shape [batch_size, 3].
               new_vecs: torch.FloatTensor   # Vectors of shape [batch_size, stack_dim].
              ) -> torch.FloatTensor:
        self.tapes = F.update_noop_stack(self.tapes, policies, new_vecs, self.max_depth)
        return self.tapes

    @classmethod
    @overrides
    def get_num_actions(cls) -> int:
        return 3
