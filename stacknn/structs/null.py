from __future__ import absolute_import

import torch
from torch.autograd import Variable

from stacknn.structs.base import Struct


class NullStruct(Struct):

    """Neural datastructure that always reads a zero vector.

    This is useful for establishing baseline performance without a
    neural datastructure.
    """

    def pop(self, strength):
        pass

    def push(self, value, strength):
        pass

    def read(self, strength):
        return Variable(torch.zeros([self.batch_size, self.embedding_size],
                        device=strength.device))
