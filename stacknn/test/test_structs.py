import unittest
import torch
from torch.autograd import Variable

from numpy.testing import assert_approx_equal

from stacknn.structs import Stack, Queue


class TestStructs(unittest.TestCase):

    def test_stack(self):
        """Stack example from Grefenstette paper."""
        stack = Stack(1, 1)
        out = stack(
            Variable(torch.FloatTensor([[1]])),
            Variable(torch.FloatTensor([[0]])),
            Variable(torch.FloatTensor([[.8]])),
        )
        stack.log()
        assert_approx_equal(out.data[0,0], .8)
        out = stack(
            Variable(torch.FloatTensor([[2]])),
            Variable(torch.FloatTensor([[.1]])),
            Variable(torch.FloatTensor([[.5]])),
        )
        stack.log()
        assert_approx_equal(out.data[0,0], 1.5)
        out = stack(
            Variable(torch.FloatTensor([[3]])),
            Variable(torch.FloatTensor([[.9]])),
            Variable(torch.FloatTensor([[.9]])),
        )
        stack.log()
        assert_approx_equal(out.data[0,0], 2.8)

    def test_queue(self):
        """Adapts example from Grefenstette paper for queues."""
        queue = Queue(1, 1)
        out = queue(
            Variable(torch.FloatTensor([[1]])),
            Variable(torch.FloatTensor([[0]])),
            Variable(torch.FloatTensor([[.8]])),
        )
        queue.log()
        assert_approx_equal(out.data[0,0], .8)
        out = queue(
            Variable(torch.FloatTensor([[2]])),
            Variable(torch.FloatTensor([[.1]])),
            Variable(torch.FloatTensor([[.5]])),
        )
        queue.log()
        assert_approx_equal(out.data[0,0], 1.3)
        out = queue(
            Variable(torch.FloatTensor([[3]])),
            Variable(torch.FloatTensor([[.9]])),
            Variable(torch.FloatTensor([[.9]])),
        )
        queue.log()
        assert_approx_equal(out.data[0,0], 2.7)


if __name__ == "__main__":
    unittest.main()
