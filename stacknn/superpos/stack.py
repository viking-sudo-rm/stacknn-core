import torch


class Stack:

    """Implements a superposition-y differentiable stack architecture inspired by Suzgun et al.,
    2019. The paper link is:
    https://arxiv.org/abs/1911.03329v1
    """

    def __init__(self, tapes: torch.Tensor):
        self.tapes = tapes
        self.device = tapes.device

    @classmethod
    def empty(cls, batch_size: int, stack_dim: int, device: int):
        tapes = torch.zeros(batch_size, 0, stack_dim, device=device)
        return cls(tapes)

    def update(self,
               policies: torch.Tensor,  # Distribution of shape [batch_size, 3].
               new_vecs: torch.Tensor   # Vectors of shape [batch_size, stack_dim].
              ):
        batch_size, length, stack_dim = self.tapes.size()

        if length == 0:
            push_tapes = new_vecs.unsqueeze(dim=1)
            noop_tapes = torch.zeros(batch_size, 1, stack_dim, device=self.device)
            pop_tapes = noop_tapes

        else:
            # Push operation.
            push_tapes = torch.empty(batch_size, length + 1, stack_dim, device=self.device)
            push_tapes[:, 0, :] = new_vecs
            push_tapes[:, 1:, :] = self.tapes

            # No operation.
            noop_tapes = torch.empty(batch_size, length + 1, stack_dim, device=self.device)
            noop_tapes[:, :-1, :] = self.tapes
            noop_tapes[:, -1, :] = 0.

            # Pop operation.
            pop_tapes = torch.empty(batch_size, length + 1, stack_dim, device=self.device)
            pop_tapes[:, :-2, :] = self.tapes[:, 1:]
            pop_tapes[:, -2:, :] = 0.

        self.tapes = policies[:, 0] * push_tapes + policies[:, 1] * noop_tapes + \
            policies[:, 2] * pop_tapes
