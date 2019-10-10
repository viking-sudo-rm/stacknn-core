from __future__ import absolute_import

from typing import List
from abc import abstractmethod

import torch
from torch.autograd import Variable
from torch.nn.functional import relu

from stacknn.structs.base import Struct


def tensor_to_string(tensor):
    """
    Formats a torch.FloatTensor as a string.

    :type tensor: torch.FloatTensor
    :param tensor: A tensor

    :rtype str
    :return: A string describing tensor
    """
    return "\t".join("{:.4f} ".format(x) for x in tensor)


def to_string(obj):
    """
    Formats a PyTorch object as a string.

    :param obj: A PyTorch object (tensor or Variable)

    :rtype: str
    :return: A string description of obj
    """
    if isinstance(obj, torch.FloatTensor):
        return tensor_to_string(obj)
    elif isinstance(obj, Variable):
        return tensor_to_string(obj.data)
    else:
        return str(obj)


def bottom_to_top(num_steps):
    return range(num_steps)


def top_to_bottom(num_steps):
    return reversed(range(num_steps))


def top(num_steps):
    return num_steps


def bottom(num_steps):
    return 0


class SimpleStruct(Struct):
    """
    Abstract class that subsumes the stack and the queue. This class is
    intended for implementing data structures that have the following
    behavior:
        - self._values is a list of vectors represented by a matrix
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

    def __init__(self, batch_size, embedding_size):
        """
        Constructor for the SimpleStruct object.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch

        :type embedding_size: int
        :param embedding_size: The size of the vectors stored in this
            SimpleStruct
        """
        super().__init__(batch_size, embedding_size)

        # Vector contents on the stack and their corresponding strengths.
        self._values: List[torch.Tensor] = []
        self._strengths: List[torch.Tensor] = []

    def __len__(self):
        return len(self._values)

    """ Struct Operations """

    @abstractmethod
    def _pop_indices(self):
        """
        Specifies the direction of the popping cascade. See self.pop for
        details on the popping operation of the SimpleStruct. This
        function should either be a generator or return an iterator.

        :rtype: Iterator
        :return: An iterator looping over indices of self._values in
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
        :return: The index of an item in self._values after it has been
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
        :return: An iterator looping over indices of self._values in
            the order of the reading cascade
        """
        raise NotImplementedError("Missing implementation for _read_indices")

    """Implement the abstract operations inherited from base Struct."""

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

        :type strength: Variable
        :param strength: The total amount of items to pop, measured by
            strength

        :return: None
        """
        pop_queue = []
        for i in self._pop_indices():
            local_strength = relu(self._strengths[i] - strength)
            strength = relu(strength - self._strengths[i])
            self._strengths[i] = local_strength

            # When we use up all our pop strength, stop.
            if (strength == 0).all():
                break

            # If this item is zero-ed, skip it and remove it.
            if (self._strengths[i] == 0).all():
                # TODO: We would keep different lists for each batch element.
                pop_queue.append(i)
                continue

        # Remove elements that are all zero.
        offset = 0
        while pop_queue:
            idx = pop_queue.pop(0) - offset
            self._values.pop(idx)
            self._strengths.pop(idx)
            offset += 1

    def push(self, value, strength):
        """
        The push operation inserts a vector and a strength somewhere in
        self._values and self._strengths. The location of the new item
        is determined by self._push_index, which gives the index of the
        new item in self._values and self._strengths after the push
        operation is complete.

        :type value: Variable
        :param value: [batch_size x embedding_size] tensor to be pushed to
        the SimpleStruct

        :type strength: Variable
        :param strength: [batch_size] tensor of strengths with which value
        will be pushed

        :return: None
        """
        push_index = self._push_index()
        self._values.insert(push_index, value)
        self._strengths.insert(push_index, strength)

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

        :type strength: float
        :param strength: The total amount of vectors to look at,
            measured by their strengths

        :rtype: Variable
        :return: The output of the read operation, described above
        """
        summary = 0.
        strength_used = 0.

        for i in self._read_indices():
            strength_weight = torch.min(self._strengths[i],
                                        relu(strength - strength_used))
            strength_weight = strength_weight.view(self.batch_size, 1)
            strength_weight = strength_weight.repeat(1, self.embedding_size)

            summary += strength_weight * self._values[i]
            strength_used = strength_used + self._strengths[i]
            if (strength_used == strength).all():
                break

        return summary

    """ Reporting """

    def print_summary(self, batch):
        """
        Prints self._values and self._strengths to the console for a
        particular batch.

        :type batch: int
        :param batch: The number of the batch to print information for

        :return: None
        """
        if batch < 0 or batch >= self.batch_size:
            raise IndexError("There is no batch {}.".format(batch))

        print("t\t|Strength\t|Value")
        print("\t|\t\t\t|")

        for t in reversed(range(len(self))):
            v_str = to_string(self._values[t][batch, :])
            s = self._strengths[t][batch].data.item()
            print("{}\t|{:.4f}\t\t|{}".format(t, s, v_str))

    def log(self):
        """
        Prints self._values and self._strengths to the console for all
        batches.

        :return: None
        """
        for b in range(self.batch_size):
            print("Batch {}:".format(b))
            self.print_summary(b)


class Stack(SimpleStruct):
    """
    A neural stack (last in, first out). Items are popped and read from
    the top of the stack to the bottom, and items are pushed to the top.
    """

    def _pop_indices(self):
        return top_to_bottom(len(self))

    def _push_index(self):
        return top(len(self))

    def _read_indices(self):
        return top_to_bottom(len(self))


class Queue(SimpleStruct):
    """
    A neural queue (first in, first out). Items are popped and read from
    top-to-bottom, and items are pushed to the bottom.
    """

    def _pop_indices(self):
        return bottom_to_top(len(self))

    def _push_index(self):
        return top(len(self))

    def _read_indices(self):
        return bottom_to_top(len(self))
