import unittest

from numpy.testing import assert_allclose
import torch
from torch.autograd import Variable

from stacknn.structs.regularization import InterfaceRegTracker, binary_reg_fn


class TestExpectation(unittest.TestCase):

    def test_simple_reg_fn(self):
        """ Test whether regularization is correctly calculated. """
        reg_fn = lambda strengths: 2 * strengths
        reg_tracker = InterfaceRegTracker(1., reg_fn=reg_fn)
        strengths = Variable(torch.ones([10]))
        reg_tracker.regularize(strengths)
        result = sum(reg_tracker.loss.data)
        assert result == 2., \
            "{} != {}".format(result, 2.)


    def test_binary_reg_fn(self):
        """ Tests whether some values of the function are correct. """
        inputs = Variable(torch.Tensor([0, .5, 1]))
        outputs = binary_reg_fn(inputs).data.numpy()
        expected = [0.0029409, 1, 0.0029409]
        assert_allclose(outputs, expected, rtol=0.00001)