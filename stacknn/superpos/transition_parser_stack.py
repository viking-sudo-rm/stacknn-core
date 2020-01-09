from overrides import overrides
import torch
from typing import Optional

from stacknn.superpos.base import AbstractStack


class TransitionParserStack(AbstractStack):

    """Implements a stack for a transition-based dependency parsing like the one used by
    Chen et al. (2014) in "A Fast and Accurate Dependency Parser Using Neural Networks".

    For an introduction, refer to https://nlp.stanford.edu/software/nndep.html.
    """

    @overrides
    def update(self,
               policies: torch.Tensor,  # Distribution of shape [batch_size, 3].
               new_vecs: torch.Tensor   # Vectors of shape [batch_size, stack_dim].
              ):
        batch_size, length, stack_dim = self.tapes.size()

        if length == 0:
            left_tapes = torch.zeros(batch_size, 1, stack_dim, device=self.device)
            right_tapes = left_tapes
            shift_tapes = new_vecs.unsqueeze(dim=1)

        else:
            # Left-Arc operation.
            left_tapes = torch.empty(batch_size, length + 1, stack_dim, device=self.device)
            if length < 2:
                left_tapes[:, :-1, :] = self.tapes
                left_tapes[:, -1, :] = 0.
            else:
                left_tapes[:, 0, :] = self.tapes[:, 0]
                left_tapes[:, 1:-2, :] = self.tapes[:, 2:]
                left_tapes[:, -2:, :] = 0.

            # Right-Arc operation.
            right_tapes = torch.empty(batch_size, length + 1, stack_dim, device=self.device)
            if length < 2:
                right_tapes[:, :-1, :] = self.tapes
                right_tapes[:, -1, :] = 0.
            else:
                right_tapes[:, :-2, :] = self.tapes[:, 1:]
                right_tapes[:, -2:, :] = 0.

            # Shift operation.
            shift_tapes = torch.empty(batch_size, length + 1, stack_dim, device=self.device)
            shift_tapes[:, 0, :] = new_vecs
            shift_tapes[:, 1:, :] = self.tapes

        pol = policies.unsqueeze(-1).unsqueeze(-1)
        self.tapes = pol[:, 0] * left_tapes + pol[:, 1] * right_tapes + pol[:, 2] * shift_tapes

    @classmethod
    @overrides
    def get_num_actions(cls):
        return 3
