import unittest
import torch
from torch.autograd import Variable

from numpy.testing import assert_approx_equal

from stacknn.superpos import RewriteStack


PUSH = torch.tensor([[1., 0., 0.]])
NOOP = torch.tensor([[0., 1., 0.]])
POP = torch.tensor([[0., 0., 1.]])

NEW_LIST = [[[1., 1., 0.]]]
NEW_VEC = torch.tensor([[1., 1., 0.]])
VEC2 = torch.tensor([[0., 0., 1.]])


class TestRewriteStack(unittest.TestCase):

    def test_initialize(self):
        stack = RewriteStack.empty(1, 3, None)
        assert stack.tapes.tolist() == [[]]

    def test_push_noop_pop(self):
        stack = RewriteStack.empty(1, 3, None)

        stack.update(PUSH, NEW_VEC)
        assert stack.tapes.tolist() == [[[1., 1., 0.]]]

        stack.update(NOOP, VEC2)
        assert stack.tapes.tolist() == [[[0., 0., 1.0], [0.0, 0.0, 0.0]]]

        stack.update(POP, NEW_VEC)
        assert stack.tapes.tolist() == [[[0., 0., 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]

    def test_pop_empty(self):
        stack = RewriteStack.empty(1, 3, None)
        stack.update(POP, NEW_VEC)
        assert stack.tapes.tolist() == [[[0., 0., 0.]]]

    def test_superpos_empty(self):
        stack = RewriteStack.empty(1, 3, None)
        policy = torch.ones(1, 3) / 3
        stack.update(policy, NEW_VEC)
        torch.testing.assert_allclose(stack.tapes.tolist(), [[[1/3, 1/3, 0]]])

    def test_complex_update(self):
        batch_size = 16
        stack_dim = 10
        policy_vectors = torch.ones(batch_size, 3)
        push_vectors = torch.ones(batch_size, stack_dim)

        stack = RewriteStack.empty(batch_size, stack_dim)
        stack.update(policy_vectors, push_vectors)

        expected = push_vectors.unsqueeze(dim=1)
        torch.testing.assert_allclose(stack.tapes, expected)

    def test_get_num_actions(self):
        assert RewriteStack.get_num_actions() == 3

    def test_returns(self):
        stack = RewriteStack.empty(1, 3, None)
        tapes = stack.update(PUSH, NEW_VEC)
        assert tapes is stack.tapes

if __name__ == "__main__":
    unittest.main()
