import unittest

import torch
from numpy.testing import assert_approx_equal

from stacknn.utils.expectation import get_expectation


class TestExpectation(unittest.TestCase):

    def test_expectation(self):
        distribution = torch.Tensor([[.2, .4, .4], [.1, .9, 0.]])
        e1, e2 = get_expectation(distribution)
        assert_approx_equal(e1, 1.2)
        assert_approx_equal(e2, .9)
