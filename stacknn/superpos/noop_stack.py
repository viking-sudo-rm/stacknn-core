from overrides import overrides
import torch
from typing import Optional

from stacknn.superpos.base import AbstractStack


class NoOpStack(AbstractStack):

    """Implements a superposition-y differentiable stack architecture inspired by Suzgun et al.,
    2019. The paper link is:
    https://arxiv.org/abs/1911.03329v1

    This stack is extended to allow no operation.
    """

    @overrides
    def update(self,
               policies: torch.Tensor,  # Distribution of shape [batch_size, 3].
               new_vecs: torch.Tensor   # Vectors of shape [batch_size, stack_dim].
              ):
        batch_size, length, stack_dim = self.tapes.size()

        if length == 0:
            push_tapes = new_vecs.unsqueeze(dim=1)
            noop_tapes = torch.zeros(batch_size, 1, stack_dim, device=self.device)
            pop_tapes = noop_tapes

        else:
            # Push operation.
            push_tapes = torch.empty(batch_size, length + 1, stack_dim, device=self.device)
            push_tapes[:, 0, :] = new_vecs
            push_tapes[:, 1:, :] = self.tapes

            # No operation.
            noop_tapes = torch.empty(batch_size, length + 1, stack_dim, device=self.device)
            noop_tapes[:, :-1, :] = self.tapes
            noop_tapes[:, -1, :] = 0.

            # Pop operation.
            pop_tapes = torch.empty(batch_size, length + 1, stack_dim, device=self.device)
            pop_tapes[:, :-2, :] = self.tapes[:, 1:]
            pop_tapes[:, -2:, :] = 0.

        policies = policies.unsqueeze(-1).unsqueeze(-1)
        self.tapes = policies[:, 0] * push_tapes + policies[:, 1] * noop_tapes + \
            policies[:, 2] * pop_tapes

    @overrides
    def get_num_actions(self):
        return 3
