import unittest
import torch
from torch.autograd import Variable

from numpy.testing import assert_approx_equal

from stacknn.superpos import Stack


PUSH = torch.tensor([[1., 0.]])
POP = torch.tensor([[0., 1.]])

NEW_LIST = [[[1., 1., 0.]]]
NEW_VEC = torch.tensor([[1., 1., 0.]])


class TestSuperStack(unittest.TestCase):

    def test_initialize(self):
        stack = Stack.empty(1, 3)
        assert stack.tapes.tolist() == [[]]

    def test_push_pop(self):
        stack = Stack.empty(1, 3)

        # Apply a push.
        stack.update(PUSH, NEW_VEC)
        assert stack.tapes.tolist() == [[[1., 1., 0.]]]

        # Apply a pop.
        stack.update(POP, NEW_VEC)
        assert stack.tapes.tolist() == [[[0., 0., 0.0], [0.0, 0.0, 0.0]]]

    def test_pop_empty(self):
        stack = Stack.empty(1, 3)
        stack.update(POP, NEW_VEC)
        assert stack.tapes.tolist() == [[[0., 0., 0.]]]

    def test_superpos_empty(self):
        stack = Stack.empty(1, 3, None)
        policy = torch.ones(1, 2) / 2
        stack.update(policy, NEW_VEC)
        torch.testing.assert_allclose(stack.tapes.tolist(), [[[1/2, 1/2, 0]]])

    def test_get_num_actions(self):
        stack = Stack.empty(1, 3, None)
        assert stack.get_num_actions() == 2

if __name__ == "__main__":
    unittest.main()
