import unittest
import torch
from torch.autograd import Variable

from numpy.testing import assert_approx_equal

from stacknn.superpos import Stack


PUSH = torch.tensor([[1., 0., 0.]])
NOOP = torch.tensor([[0., 1., 0.]])
POP = torch.tensor([[0., 0., 1.]])

NEW_LIST = [[[1., 1., 0.]]]
NEW_VEC = torch.tensor([[1., 1., 0.]])


class TestSuperStack(unittest.TestCase):

    def test_initialize(self):
        stack = Stack.empty(1, 3, None)
        assert stack.tapes.tolist() == [[]]

    def test_push_noop_pop(self):
        stack = Stack.empty(1, 3, None)

        # Apply a push.
        stack.update(PUSH, NEW_VEC)
        assert stack.tapes.tolist() == [[[1., 1., 0.]]]

        # Apply a no-op.
        stack.update(NOOP, NEW_VEC)
        assert stack.tapes.tolist() == [[[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]

        # Apply a pop.
        stack.update(POP, NEW_VEC)
        assert stack.tapes.tolist() == [[[0., 0., 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]

    def test_pop_empty(self):
        stack = Stack.empty(1, 3, None)
        stack.update(POP, NEW_VEC)
        assert stack.tapes.tolist() == [[[0., 0., 0.]]]

    def test_superpos_empty(self):
        stack = Stack.empty(1, 3, None)
        policy = torch.ones(1, 3) / 3
        stack.update(policy, NEW_VEC)
        torch.testing.assert_allclose(stack.tapes.tolist(), [[[1/3, 1/3, 0]]])


if __name__ == "__main__":
    unittest.main()
