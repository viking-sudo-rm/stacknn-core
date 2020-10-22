from overrides import overrides
import torch
from typing import Optional

from .base import AbstractStack
from . import functional as F


class MultiPopStack(AbstractStack):

    """Implements a multipop superpositiony stack extending Suzgun et al. (2019). This architecture
    also bears resemblance to the architecture proposed by Yogatama et al.

    The stack utilizes reduce-k operations which combine k pops with one push.
    """

    def __init__(self,
                 stack_dim: int,
                 max_depth: Optional[int] = None,
                 num_actions: Optional[int] = 6):
        super().__init__(stack_dim, max_depth)
        self.num_actions = num_actions

    @overrides
    def update(self,
               policies: torch.FloatTensor,  # Distribution of shape [batch_size, num_actions].
               new_vecs: torch.FloatTensor   # Vectors of shape [batch_size, stack_dim].
              ) -> torch.FloatTensor:
        self.tapes = F.update_kpop_stack(self.tapes, policies, new_vecs, self.num_actions, self.max_depth)
        return self.tapes

    @overrides
    def get_num_actions(self) -> int:
        return self.num_actions
