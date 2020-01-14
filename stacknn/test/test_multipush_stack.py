import unittest
import torch
from torch.autograd import Variable

from numpy.testing import assert_approx_equal

from stacknn.superpos import MultiPushStack


NUM_ACTIONS = 6
PUSH0 = torch.tensor([[1., 0., 0., 0., 0., 0.]])
PUSH2 = torch.tensor([[0., 0., 1., 0., 0., 0.]])
PUSH3 = torch.tensor([[0., 0., 0., 1., 0., 0.]])
PUSH5 = torch.tensor([[0., 0., 0., 0., 0., 1.]])

NEW_VEC = torch.tensor([[1., 1., 0.]])


class TestMultiPushStack(unittest.TestCase):

    def test_initialize(self):
        stack = MultiPushStack.empty(1, 3)
        assert stack.tapes.tolist() == [[]]

    def test_push_pop(self):
        stack = MultiPushStack.empty(1, 3)

        stack.update(PUSH3, NEW_VEC)
        assert stack.tapes.tolist() == [[[1., 1., 0.],
                                         [1., 1., 0.],
                                         [1., 1., 0.],
                                         [0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0]]]

        stack.update(PUSH5, 2 * NEW_VEC)
        assert stack.tapes.tolist() == [[[2., 2., 0.],
                                         [2., 2., 0.],
                                         [2., 2., 0.],
                                         [2., 2., 0.],
                                         [2., 2., 0.],
                                         [1., 1., 0.],
                                         [1., 1., 0.],
                                         [0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0]]]

    def test_pop_empty(self):
        stack = MultiPushStack.empty(1, 3)
        stack.update(PUSH0, NEW_VEC)
        assert stack.tapes.tolist() == [[[0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0]]]

    def test_superpos(self):
        stack = MultiPushStack.empty(1, 3)
        stack.update(PUSH2, NEW_VEC)
        policy = torch.tensor([[1/2, 1/2, 0, 0, 0, 0]])
        stack.update(policy, NEW_VEC)

        # Should have .5 * [1] + .5 * [1, 1] = [1, .5].
        expected = [[
            [1., 1., 0.],
            [.5, .5, 0.],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]]
        torch.testing.assert_allclose(stack.tapes.tolist(), expected)

    def test_get_num_actions(self):
        assert MultiPushStack.get_num_actions() == NUM_ACTIONS

if __name__ == "__main__":
    unittest.main()
