from distutils.core import setup


setup(name="stacknn",
      version="0.2",
      description="Differentiable stacks and queues in PyTorch",
      author="Will Merrill, Computational Linguistics at Yale",
      url="https://github.com/viking-sudo-rm/StackNN",
      packages=["stacknn", "stacknn.structs", "stacknn.utils"],
)
