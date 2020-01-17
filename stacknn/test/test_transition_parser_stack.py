import unittest
import torch
from torch.autograd import Variable

from numpy.testing import assert_approx_equal

from stacknn.superpos import TransitionParserStack


LEFT = torch.tensor([[1., 0., 0.]])
RIGHT = torch.tensor([[0., 1., 0.]])
SHIFT = torch.tensor([[0., 0., 1.]])

VEC1 = torch.tensor([[1., 0., 0., 0.]])
VEC2 = torch.tensor([[0., 1., 0., 0.]])


class TestTransitionParserStack(unittest.TestCase):

    def test_initialize(self):
        stack = TransitionParserStack.empty(1, 4)
        assert stack.tapes.tolist() == [[]]

    def test_shift_shift_left(self):
        stack = TransitionParserStack.empty(1, 4)
        stack.update(SHIFT, VEC1)
        assert stack.tapes.tolist() == [[[1., 0., 0., 0.]]]
        stack.update(SHIFT, VEC2)
        assert stack.tapes.tolist() == [[[0., 1., 0., 0.], [1., 0., 0., 0.]]]
        stack.update(LEFT, VEC1)
        assert stack.tapes.tolist() == [[[0., 1., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]]

    def test_shift_shift_right(self):
        stack = TransitionParserStack.empty(1, 4)
        stack.update(SHIFT, VEC1)
        assert stack.tapes.tolist() == [[[1., 0., 0., 0.]]]
        stack.update(SHIFT, VEC2)
        assert stack.tapes.tolist() == [[[0., 1., 0., 0.], [1., 0., 0., 0.]]]
        stack.update(RIGHT, VEC1)
        assert stack.tapes.tolist() == [[[1., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]]

    def test_empty(self):
        stack = TransitionParserStack.empty(1, 4)
        stack.update(LEFT, VEC1)
        assert stack.tapes.tolist() == [[[0., 0., 0., 0.]]]
        stack = TransitionParserStack.empty(1, 4)
        stack.update(RIGHT, VEC1)
        assert stack.tapes.tolist() == [[[0., 0., 0., 0.]]]

    def test_superpos_empty(self):
        stack = TransitionParserStack.empty(1, 4, None)
        policy = torch.ones(1, 3) / 3
        stack.update(SHIFT, VEC1)
        stack.update(SHIFT, VEC2)
        stack.update(policy, VEC1)
        expected = [[
            [2/3, 1/3, 0, 0],
            [0, 1/3, 0, 0],
            [1/3, 0, 0, 0],
        ]]
        torch.testing.assert_allclose(stack.tapes.tolist(), expected)

    def test_get_num_actions(self):
        assert TransitionParserStack.get_num_actions() == 3

    def test_returns(self):
        stack = TransitionParserStack.empty(1, 4, None)
        tapes = stack.update(SHIFT, VEC1)
        assert tapes is stack.tapes

if __name__ == "__main__":
    unittest.main()
