# `stacknn-core`: The Successor to StackNN
This library implements various types of differentiable stacks and queues in PyTorch. In stacknn/structs, we include our original *weighted* stack implementation, and in stacknn/superpos, we implement several kinds of *superimposed* differentiable stacks used by [Suzgun et al. (2019)](https://arxiv.org/abs/1911.03329v1) and others.

## Weighted Stack

The weighted stack in this library is a light-weight version of the one in [StackNN](https://github.com/viking-sudo-rm/StackNN) that is easier to install and optimized for faster training. The API is also straightforward. For example, to construct a differentiable stack and perform a push, all you have to do is:

```python
from stacknn.structs import Stack
stack = Stack(BATCH_SIZE, STACK_DIM)
read_vectors = stack(value_vectors, pop_strengths, push_strengths)
```

For more complex use cases, refer to the (old) [StackNN](https://github.com/viking-sudo-rm/StackNN) or [industrial-stacknns](https://github.com/viking-sudo-rm/industrial-stacknns) repositories.

The weighted stack is associated with the paper [Context-Free Transductions with Neural Stacks](https://arxiv.org/abs/1809.02836), which appeared at the Analyzing and Interpreting Neural Networks for NLP workshop at EMNLP 2018. Refer to our paper for more theoretical background on differentiable data structures.

## Superimposed Stack

The architecture in this subpackage is based on the one used by [Suzgun et al. (2019)](https://arxiv.org/abs/1911.03329v1). Example usage:

```python
from stacknn.superpos import Stack
stack = Stack.empty(BATCH_SIZE, STACK_DIM)
stack.update(policy_vectors, value_vectors)
stack.tapes  # Returns a [batch_size, depth, STACK_DIM] tensor of the stack contents.
```

The superposition-based stack framework allows for many different variants. We implement many of these in `stacknn.superpos`. Since v0.9.3, superposition stacks also support an immutable paradigm recalling functional programming. For example:

```python
import stacknn.superpos.functional as F
tapes = torch.zeros(BATCH_SIZE, 0, STACK_DIM)
new_tapes = F.update_stack(tapes, policy_vectors, value_vectors)
```

## Installation

```shell
pip install git+https://github.com/viking-sudo-rm/stacknn-core
```

The library only supports Python 3 and depends on numpy and torch.

## Contributing

We welcome contributions from outside in the form of pull requests. Please report any bugs in the GitHub issues tracker. If you are a Yale student interested in joining [Computational Linguistics at Yale](http://clay.yale.edu/) for this or another project, please contact Bob Frank.

## Citations

If you use the weighted stack in your research, please cite the following paper. If you use the superposition-based stack, please cite [Suzgun et al. (2019)](https://arxiv.org/abs/1911.03329v1).

```
@inproceedings{hao-etal-2018-context,
    title = "Context-Free Transductions with Neural Stacks",
    author = "Hao, Yiding  and
      Merrill, William  and
      Angluin, Dana  and
      Frank, Robert  and
      Amsel, Noah  and
      Benz, Andrew  and
      Mendelsohn, Simon",
    booktitle = "Proceedings of the 2018 {EMNLP} Workshop {B}lackbox{NLP}: Analyzing and Interpreting Neural Networks for {NLP}",
    month = nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W18-5433",
    pages = "306--315",
    abstract = "This paper analyzes the behavior of stack-augmented recurrent neural network (RNN) models. Due to the architectural similarity between stack RNNs and pushdown transducers, we train stack RNN models on a number of tasks, including string reversal, context-free language modelling, and cumulative XOR evaluation. Examining the behavior of our networks, we show that stack-augmented RNNs can discover intuitive stack-based strategies for solving our tasks. However, stack RNNs are more difficult to train than classical architectures such as LSTMs. Rather than employ stack-based strategies, more complex stack-augmented networks often find approximate solutions by using the stack as unstructured memory.",
}
```

## Acknowledgements

Thanks to the various members of [Computational Linguistics at Yale](http://clay.yale.edu/) who contributed to the various iterations of this library. All the contributors are listed on the Contributors page.

## Unit Tests

To run the unit tests for this library, execute the follow command from the root stacknn-core directory:
```shell
pytest stacknn/
```
