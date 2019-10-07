from __future__ import print_function, division

import torch
from torch.autograd import Variable


# Useful for debugging. Make sure it is larger than test set.
_MAX_COUNT = 100000


def binary_reg_fn(strengths):
    """ Function that is low around 0 and 1. """
    term = 3.25 * strengths - 1.625
    return 1 / (1 + torch.pow(term, 12))


class InterfaceRegTracker:
    """
    Compute arbitrary regularization function on struct interface.
    """

    def __init__(self, reg_weight, reg_fn=binary_reg_fn):
        """
        Constructor for StructInterfaceLoss.

        :type reg_weight: float
        :param reg_weight: Linear weight for regularization loss.

        :type reg_fn: function
        :param reg_fn: Regularization function to apply over 1D tensor

        """
        self._reg_weight = reg_weight
        self._reg_fn = reg_fn
        self._loss = Variable(torch.zeros([1]))
        self._count = 0

    @property
    def reg_weight(self):
        return self._reg_weight
    
    @property
    def loss(self):
        return self._reg_weight * self._loss / self._count

    def regularize(self, strengths):
        assert self._count < _MAX_COUNT, \
            "Max regularization count exceeded. Are you calling reg_tracker.reset()?"
        losses = self._reg_fn(strengths)
        self._loss += torch.sum(losses)
        self._count += len(losses)

    def reset(self):
        self._loss = Variable(torch.zeros([1]))
        self._count = 0
