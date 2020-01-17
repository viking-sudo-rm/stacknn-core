import unittest
import torch
from torch.autograd import Variable

from numpy.testing import assert_approx_equal

from stacknn.superpos import MinimalistStack


PUSH = torch.tensor([[1., 0.]])
MERGE = torch.tensor([[0., 1.]])

VEC1 = torch.tensor([[1., 1., 0.]])
VEC2 = torch.tensor([[0., 1., 1.]])


class TestSuperStack(unittest.TestCase):

    def test_initialize(self):
        stack = MinimalistStack.empty(1, 3)
        assert stack.tapes.tolist() == [[]]

    def test_push2_merge(self):
        stack = MinimalistStack.empty(1, 3)
        stack.update(PUSH, VEC1)
        assert stack.tapes.tolist() == [[[1., 1., 0.]]]
        stack.update(PUSH, VEC1)
        assert stack.tapes.tolist() == [[[1., 1., 0.], [1., 1., 0.]]]
        stack.update(MERGE, VEC2)
        assert stack.tapes.tolist() == [[[0., 1., 1.], [0., 0., 0.0], [0.0, 0.0, 0.0]]]

    def test_pop_empty(self):
        stack = MinimalistStack.empty(1, 3)
        stack.update(MERGE, VEC1)
        assert stack.tapes.tolist() == [[[1., 1., 0.]]]

    def test_superpos(self):
        stack = MinimalistStack.empty(1, 3)
        stack.update(PUSH, VEC1)
        stack.update(PUSH, VEC1)
        policy = torch.ones(1, 2) / 2
        stack.update(policy, VEC1)
        expected = [[[1., 1., 0], [1/2, 1/2, 0], [1/2, 1/2, 0]]]
        torch.testing.assert_allclose(stack.tapes.tolist(), expected)

    def test_get_num_actions(self):
        assert MinimalistStack.get_num_actions() == 2

    def test_returns(self):
        stack = MinimalistStack.empty(1, 3, None)
        tapes = stack.update(PUSH, VEC1)
        assert tapes is stack.tapes

if __name__ == "__main__":
    unittest.main()
