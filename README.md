# `stacknn-core`: The Successor to StackNN
This library implements differentiable stacks and queues in PyTorch. It is a light-weight version of [StackNN](https://github.com/viking-sudo-rm/StackNN) that is easier to install and integrate with any framework. For example, to construct a differentiable stack and perform a push, all you have to do is:

```python
from stacknn.structs import Stack
stack = Stack(BATCH_SIZE, STACK_VECTOR_SIZE)
read_vectors = stack(value_vectors, pop_strengths, push_strengths)
```

For more complex use cases, refer to the (old) [StackNN](https://github.com/viking-sudo-rm/StackNN) or [industrial-stacknns](https://github.com/viking-sudo-rm/industrial-stacknns) repositories.

All the code in this repository is associated with the paper [Context-Free Transductions with Neural Stacks](https://arxiv.org/abs/1809.02836), which appeared at the Analyzing and Interpreting Neural Networks for NLP workshop at EMNLP 2018. Refer to our paper for more theoretical background on differentiable data structures.

## Installation

```shell
pip install git+https://github.com/viking-sudo-rm/stacknn-core
```

Depends on numpy and torch.

## Contributing

This project is managed by [Computational Linguistics at Yale](http://clay.yale.edu/). We welcome contributions from outside in the form of pull requests. Please report any bugs in the GitHub issues tracker. If you are a Yale student interested in joining our lab, please contact Bob Frank.

## Citations

If you use this codebase in your research, please cite the associated paper:

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
