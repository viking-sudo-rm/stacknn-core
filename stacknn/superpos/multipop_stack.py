from overrides import overrides
import torch
from typing import Optional

from stacknn.superpos.base import AbstractStack


class MultiPopStack(AbstractStack):

    """Implements a multipop superpositiony stack extending Suzgun et al. (2019). This architecture
    also bears resemblance to the architecture proposed by Yogatama et al.

    The stack utilizes reduce-k operations which combine k pops with one push.
    """

    def __init__(self, stack_dim: int, num_actions: int = 6):
        super().__init__(stack_dim)
        self.num_actions = num_actions

    @overrides
    def update(self,
               policies: torch.Tensor,  # Distribution of shape [batch_size, num_actions].
               new_vecs: torch.Tensor   # Vectors of shape [batch_size, stack_dim].
              ) -> None:
        batch_size, length, stack_dim = self.tapes.size()

        policies = policies.unsqueeze(-1).unsqueeze(-1)
        tapes = torch.empty(batch_size, self.num_actions, length + 1, stack_dim,
                            device=self.device)
        for action in range(self.num_actions):
            tapes[:, action, 0, :] = new_vecs

            if action <= length:
                # Remove action-many elements from the stack.
                tapes[:, action, 1:1 + length - action, :] = self.tapes[:, :length - action, :]
                tapes[:, action, 1 + length - action:, :] = 0.
            else:
                # Remove everything from the stack.
                tapes[:, action, 1:, :] = 0.

        tapes = policies * tapes
        self.tapes = torch.sum(tapes, dim=1)

    @overrides
    def get_num_actions(self):
        return self.num_actions
