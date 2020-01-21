from overrides import overrides
import torch
from typing import Optional

from stacknn.superpos.base import AbstractStack


class RewriteStack(AbstractStack):

    """This stack is similar to the NoOp stack, but NoOp is replaced by an operation that rewrites
    the top element of the stack.
    """

    @overrides
    def update(self,
               policies: torch.FloatTensor,  # Distribution of shape [batch_size, 3].
               new_vecs: torch.FloatTensor   # Vectors of shape [batch_size, stack_dim].
              ) -> torch.FloatTensor:
        batch_size, length, stack_dim = self.tapes.size()

        if length == 0:
            push_tapes = new_vecs.unsqueeze(dim=1)
            rewrite_tapes = torch.zeros(batch_size, 1, stack_dim, device=self.device)
            pop_tapes = rewrite_tapes

        else:
            # Push operation.
            push_tapes = torch.empty(batch_size, length + 1, stack_dim, device=self.device)
            push_tapes[:, 0, :] = new_vecs
            push_tapes[:, 1:, :] = self.tapes

            # Rewrite operation.
            rewrite_tapes = torch.empty(batch_size, length + 1, stack_dim, device=self.device)
            rewrite_tapes[:, 0, :] = new_vecs
            rewrite_tapes[:, 1:-1, :] = self.tapes[:, 1:, :]
            rewrite_tapes[:, -1, :] = 0.

            # Pop operation.
            pop_tapes = torch.empty(batch_size, length + 1, stack_dim, device=self.device)
            pop_tapes[:, :-2, :] = self.tapes[:, 1:]
            pop_tapes[:, -2:, :] = 0.

        policies = policies.unsqueeze(-1).unsqueeze(-1)
        self.tapes = policies[:, 0] * push_tapes + policies[:, 1] * rewrite_tapes + \
            policies[:, 2] * pop_tapes

        self._enforce_max_depth()
        return self.tapes

    @classmethod
    @overrides
    def get_num_actions(cls) -> int:
        return 3
