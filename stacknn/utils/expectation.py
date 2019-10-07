from __future__ import print_function

from numpy.testing import assert_approx_equal

import torch


def get_expectation(distribution):
    """Take the row-wise expectation : [batch_size, num_options] -> [batch_size]."""
    values = torch.arange(distribution.size(1), device=distribution.device)
    values = values.unsqueeze(1)
    expectation = torch.mm(distribution, values.float())
    return expectation.squeeze(1)
