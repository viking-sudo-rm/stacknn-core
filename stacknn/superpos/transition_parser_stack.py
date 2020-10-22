from overrides import overrides
import torch
from typing import Optional

from .base import AbstractStack
from . import functional as F


class TransitionParserStack(AbstractStack):

    """Implements a stack for a transition-based dependency parsing like the one used by
    Chen et al. (2014) in "A Fast and Accurate Dependency Parser Using Neural Networks".

    For an introduction, refer to https://nlp.stanford.edu/software/nndep.html.
    """

    @overrides
    def update(self,
               policies: torch.FloatTensor,  # Distribution of shape [batch_size, 3].
               new_vecs: torch.FloatTensor   # Vectors of shape [batch_size, stack_dim].
              ) -> torch.FloatTensor:
        self.tapes = F.update_transition_parser_stack(self.tapes, policies, new_vecs, self.max_depth)
        return self.tapes

    @classmethod
    @overrides
    def get_num_actions(cls) -> int:
        return 3
