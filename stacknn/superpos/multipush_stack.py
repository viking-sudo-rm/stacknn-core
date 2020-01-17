from overrides import overrides
import torch
from typing import Optional

from stacknn.superpos.base import AbstractStack


class MultiPushStack(AbstractStack):

    """Implements a stack where one item is always popped, and then multiple vectors are pushed.

    This stack is potentially better for head-initial languages/for helping the controller overcome
    its inductive bias to operate recurrently.

    TODO: We could improve the efficiency of this implementation by changing how underlying data is
    represented. Associate a "weight" with each item on the stack.
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
        batch_size, length, stack_dim = self.tapes.size()
        new_vecs = new_vecs.unsqueeze(1)
        policies = policies.unsqueeze(-1).unsqueeze(-1)
        tapes = torch.empty(batch_size, self.num_actions, length + self.num_actions, stack_dim,
                            device=self.device)

        for action in range(self.num_actions):
            stacked_new_vecs = new_vecs.repeat(1, action, 1)
            tapes[:, action, :action, :] = stacked_new_vecs

            if length > 0:
                cutoff = action + length - 1
                tapes[:, action, action:cutoff, :] = self.tapes[:, 1:, :]
                tapes[:, action, cutoff:, :] = 0.
            else:
                tapes[:, action, action:, :] = 0.

        tapes = policies * tapes
        self.tapes = torch.sum(tapes, dim=1)

        self._enforce_max_depth()
        return self.tapes

    @overrides
    def get_num_actions(self) -> int:
        return self.num_actions
