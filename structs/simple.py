"""
Implementation of simple neural data structures, including the neural
stack and the neural queue.
"""
from abc import ABCMeta, abstractmethod

import torch
from torch.autograd import Variable
from torch.nn.functional import relu

from base import Struct


def tensor_to_string(tensor):
    """
    Formats a torch.FloatTensor as a string.

    :type tensor: torch.FloatTensor
    :param tensor: A tensor

    :rtype str
    :return: A string describing tensor
    """
    return "\t".join("{:.4f} ".format(x) for x in tensor)


def first_to_last(num_steps):
    return xrange(num_steps)


def last_to_first(num_steps):
    return reversed(xrange(num_steps))


def top(num_steps):
    return num_steps


def bottom(num_steps):
    return 0


class SimpleStruct(Struct):
    """
    Abstract class that subsumes the stack and the queue. This class is
    intended for implementing data structures that have the following
    behavior:
        - self.contents is a list of vectors represented by a matrix
        - popping consists of removing items from the structure in a
            cascading fashion
        - pushing consists of inserting an item at some position in the
            list of vectors
        - reading consists of taking the average of a cascade of items,
            weighted by their strengths.

    To use this class, the user must override self._pop_indices,
    self._push_index, and self_read_indices. Doing so specifies the
    direction of the popping and reading cascades, as well as the
    position in which pushed items are inserted. See Stack and Queue
    below for examples.
    """
    __metaclass__ = ABCMeta

    def __init__(self, batch_size, embedding_size, k=None):
        """
        Constructor for the SimpleStruct object.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch

        :type embedding_size: int
        :param embedding_size: The size of the vectors stored in this
            SimpleStruct
        """
        super(SimpleStruct, self).__init__(batch_size, embedding_size)
        self._t = 0

    @abstractmethod
    def _pop_indices(self):
        """
        Specifies the direction of the popping cascade. See self.pop for
        details on the popping operation of the SimpleStruct. This
        function should either be a generator or return an iterator.

        :rtype: Iterator
        :return: An iterator looping over indices of self.contents in
            the order of the popping cascade
        """
        raise NotImplementedError("Missing implementation for _pop_indices")

    @abstractmethod
    def _push_index(self):
        """
        Specifies the location where a pushed item is inserted. See
        self.push for details on the pushing operation of the
        SimpleStruct.

        :rtype: int
        :return: The index of an item in self.contents after it has been
            pushed to the SimpleStruct
        """
        raise NotImplementedError("Missing implementation for _push_index")

    @abstractmethod
    def _read_indices(self):
        """
        Specifies the direction of the reading cascade. See self.read
        for details on the reading operation of the SimpleStruct. This
        function should either be a generator or return an iterator.

        :rtype: Iterator
        :return: An iterator looping over indices of self.contents in
            the order of the reading cascade
        """
        raise NotImplementedError("Missing implementation for _read_indices")

    def pop(self, strength):
        """
        Popping is done by decreasing the strength of items in the
        SimpleStruct until they reach a strength of 0. The pop operation
        begins with an amount of strength specified by the strength
        parameter, and this amount is "consumed" such that the total
        amount of strength subtracted is equal to the initial amount of
        strength. When an item reaches a strength of 0, but the amoount
        of remaining strength is greater than 0, the remaining strength
        is used to decrease the strength of the next item. The order in
        which the items are popped is determined by self._pop_indices.

        :type strength: float
        :param strength: The total amount of items to pop, measured by
            strength

        :return: None
        """
        u = strength
        s = Variable(torch.FloatTensor(self._t, self.batch_size))
        for i in self._pop_indices():
            s_i = relu(self.strengths[i, :] - u)
            u = relu(u - self.strengths[i, :])
            s[i, :] = s_i
            # TODO: Figure out a way to break early
        self.strengths = s

        return

    def push(self, value, strength):
        """
        The push operation inserts a vector and a strength somewhere in
        self.contents and self.strengths. The location of the new item
        is determined by self._push_index, which gives the index of the
        new item in self.contents and self.strengths after the push
        operation is complete.

        :type value: torch.FloatTensor
        :param value: The vector to be pushed to the SimpleStruct

        :type strength: Variable
        :param strength: The strength with which value will be pushed

        :return: None
        """
        v = value.view(1, self.batch_size, self.embedding_size)
        s = Variable(torch.ones(1, self.batch_size) * strength.data)
        if self._t == 0:
            self.contents = v
            self.strengths = s
        else:
            i = self._push_index()
            if i == 0:
                self.contents = torch.cat([v, self.contents], 0)
                self.strengths = torch.cat([s, self.strengths], 0)
            elif i == self._t:
                self.contents = torch.cat([self.contents, v], 0)
                self.strengths = torch.cat([self.strengths, s], 0)
            else:
                first_v = self.contents[:i, :, :]
                first_s = self.strengths[:i, :, :]
                last_v = self.contents[i:, :, :]
                last_s = self.strengths[i:, :, :]

                self.contents = torch.cat([first_v, v, last_v], 0)
                self.strengths = torch.cat([first_s, s, last_s], 0)

        self._t += 1
        return

    def read(self, strength):
        """
        The read operation looks at the first few items on the stack, in
        the order determined by self._read_indices, such that the total
        strength of these items is equal to the value of the strength
        parameter. If necessary, the strength of the last vector is
        reduced so that the total strength of the items read is exactly
        equal to the strength parameter. The output of the read
        operation is computed by taking the sum of all the vectors
        looked at, weighted by their strengths.

        :type strength: torch.FloatTensor
        :param strength: The total amount of vectors to look at,
            measured by their strengths

        :rtype: Variable
        :return: The output of the read operation, described above
        """
        r = Variable(torch.zeros([self.batch_size, self.embedding_size]))
        s = Variable(torch.ones(self.batch_size) * strength)
        for i in self._read_indices():
            s_i = torch.min(self.strengths[i, :], relu(s))
            s = relu(s - s_i)
            s_i = s_i.view(self.batch_size, 1).data
            r += Variable(s_i.repeat(1, self.embedding_size) *
                          self.contents[i, :, :])

        return r

    def print_summary(self, batch):
        """
        Prints self.contents and self.strengths to the console for a
        particular batch.

        :type batch: int
        :param batch: The number of the batch to print information for

        :return: None
        """
        if batch < 0 or batch >= self.batch_size:
            raise IndexError("There is no batch {}.".format(batch))

        print "t\t|Strength\t|Value"
        print "\t|\t\t\t|"

        for t in reversed(xrange(self._t)):
            v_str = tensor_to_string(self.contents[t, batch, :])
            s = self.strengths[t, batch].data[0]
            print "{}\t|{:.4f}\t\t|{}".format(t, s, v_str)

    def log(self):
        """
        Prints self.contents and self.strengths to the console for all
        batches.

        :return: None
        """
        for b in xrange(self.batch_size):
            print "Batch {}:".format(b)
            self.print_summary(b)


class Stack(SimpleStruct):
    """
    A neural stack (last in, first out). Items are popped and read from
    the top of the stack to the bottom, and items are pushed to the top.
    """

    def _pop_indices(self):
        return last_to_first(self._t)

    def _push_index(self):
        return top(self._t)

    def _read_indices(self):
        return last_to_first(self._t)


class Queue(SimpleStruct):
    """
    A neural queue (first in, first out). Items are popped and read from
    top-to-bottom, and items are pushed to the bottom.
    """

    def _pop_indices(self):
        return last_to_first(self._t)

    def _push_index(self):
        return bottom(self._t)

    def _read_indices(self):
        return last_to_first(self._t)