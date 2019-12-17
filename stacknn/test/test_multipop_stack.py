import unittest
import torch
from torch.autograd import Variable

from numpy.testing import assert_approx_equal

from stacknn.superpos import MultiPopStack


NUM_ACTIONS = 6
REDUCE0 = torch.tensor([[1., 0., 0., 0., 0., 0.]])
REDUCE1 = torch.tensor([[0., 1., 0., 0., 0., 0.]])
REDUCE2 = torch.tensor([[0., 0., 1., 0., 0., 0.]])


NEW_LIST = [[[1., 1., 0.]]]
NEW_VEC = torch.tensor([[1., 1., 0.]])


class TestMultipopStack(unittest.TestCase):

    def test_initialize(self):
        stack = MultiPopStack.empty(1, 3)
        assert stack.tapes.tolist() == [[]]

    def test_push_pop(self):
        stack = MultiPopStack.empty(1, 3)

        # Apply a push.
        stack.update(REDUCE0, NEW_VEC)
        assert stack.tapes.tolist() == NEW_LIST

        # Apply a pop.
        stack.update(REDUCE1, torch.tensor([[1., 0., 1.]]))
        assert stack.tapes.tolist() == [[[1., 0., 1.], [0.0, 0.0, 0.0]]]

    def test_superpos(self):
        stack = MultiPopStack.empty(1, 3, None)
        stack.update(REDUCE0, NEW_VEC)
        policy = torch.tensor([[1/2, 1/2, 0, 0, 0, 0]])
        stack.update(policy, NEW_VEC)
        expected = [[[1., 1., 0.], [1/2, 1/2, 0]]]
        torch.testing.assert_allclose(stack.tapes.tolist(), expected)

    def test_get_num_actions(self):
        assert MultiPopStack.get_num_actions() == NUM_ACTIONS

if __name__ == "__main__":
    unittest.main()
