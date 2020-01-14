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

    def __init__(self, stack_dim: int, num_actions: int = 6):
        super().__init__(stack_dim)
        self.num_actions = num_actions

    @overrides
    def update(self,
               policies: torch.Tensor,  # Distribution of shape [batch_size, num_actions].
               new_vecs: torch.Tensor   # Vectors of shape [batch_size, stack_dim].
              ):
        batch_size, length, stack_dim = self.tapes.size()

        new_vecs = new_vecs.unsqueeze(1)
        policies = policies.unsqueeze(-1).unsqueeze(-1)
        weighted_tapes = []
        for action in range(self.num_actions):
            tapes = torch.empty(batch_size, length + self.num_actions, stack_dim,
                                device=self.device)
            stacked_new_vecs = new_vecs.repeat(1, action, 1)
            tapes[:, :action, :] = stacked_new_vecs

            if length > 0:
                cutoff = action + length - 1
                tapes[:, action:cutoff, :] = self.tapes[:, 1:, :]
                tapes[:, cutoff:, :] = 0.
            else:
                tapes[:, action:] = 0.

            # TODO: Can refactor this part as a matrix multiplication.
            weighted_tape = policies[:, action] * tapes
            weighted_tapes.append(weighted_tape)

        self.tapes = sum(weighted_tapes)

    @overrides
    def get_num_actions(self):
        return self.num_actions
