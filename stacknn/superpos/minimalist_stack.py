from overrides import overrides
import torch
from typing import Optional

from stacknn.superpos.base import AbstractStack
import stacknn.superpos.functional as F


class MinimalistStack(AbstractStack):

    """This stack can be used to implement a differentiable minimalist grammar formalism.

    TODO: Add the BIND operation in addition to merge.
    """

    @overrides
    def update(self,
               policies: torch.FloatTensor,  # Distribution of shape [batch_size, 2].
               new_vecs: torch.FloatTensor   # Vectors of shape [batch_size, stack_dim].
              ) -> torch.FloatTensor:
        self.tapes = F.update_minimalist_stack(self.tapes, policies, new_vecs, self.max_depth)
        return self.tapes

    @classmethod
    @overrides
    def get_num_actions(cls) -> int:
        return 2
