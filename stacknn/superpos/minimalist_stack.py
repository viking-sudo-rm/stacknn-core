from overrides import overrides
import torch
from typing import Optional

from stacknn.superpos.base import AbstractStack


class MinimalistStack(AbstractStack):

    """This stack can be used to implement a differentiable minimalist grammar formalism.

    TODO: Add the BIND operation in addition to merge.
    """

    @overrides
    def update(self,
               policies: torch.FloatTensor,  # Distribution of shape [batch_size, 2].
               new_vecs: torch.FloatTensor   # Vectors of shape [batch_size, stack_dim].
              ) -> torch.FloatTensor:
        batch_size, length, stack_dim = self.tapes.size()

        # Push operation.
        if length == 0:
            push_tapes = new_vecs.unsqueeze(dim=1)
        else:
            push_tapes = torch.empty(batch_size, length + 1, stack_dim, device=self.device)
            push_tapes[:, 0, :] = new_vecs
            push_tapes[:, 1:, :] = self.tapes

        # Merge operation.
        merge_tapes = torch.empty(batch_size, length + 1, stack_dim, device=self.device)
        merge_tapes[:, 0, :] = new_vecs
        if length > 2:
            merge_tapes[:, 1:-2, :] = self.tapes[:, 2:]
            merge_tapes[:, -2:, :] = 0.
        else:
            merge_tapes[:, 1:, :] = 0.


        policies = policies.unsqueeze(-1).unsqueeze(-1)
        self.tapes = policies[:, 0] * push_tapes + policies[:, 1] * merge_tapes

        self._enforce_max_depth()
        return self.tapes

    @classmethod
    @overrides
    def get_num_actions(cls) -> int:
        return 2
