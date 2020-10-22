from overrides import overrides
import torch
from typing import Optional

from stacknn.superpos.base import AbstractStack
import stacknn.superpos.functional as F


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
        self.tapes = F.update_kpush_stack(self.tapes, policies, new_vecs, self.num_actions, self.max_depth)
        return self.tapes

    @overrides
    def get_num_actions(self) -> int:
        return self.num_actions
