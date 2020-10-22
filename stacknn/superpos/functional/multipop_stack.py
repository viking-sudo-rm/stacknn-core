from typing import Optional
import torch

from .base import enforce_max_depth


def update_kpop_stack(tapes: torch.FloatTensor,
                      policies: torch.FloatTensor,  # Distribution of shape [batch_size, num_actions].
                      new_vecs: torch.FloatTensor,  # Vectors of shape [batch_size, stack_dim].
                      num_actions: int,
                      max_depth: Optional[int] = None,
                     ) -> torch.FloatTensor:
        batch_size, length, stack_dim = tapes.size()
        device = tapes.device

        policies = policies.unsqueeze(-1).unsqueeze(-1)
        new_tapes = torch.empty(batch_size, num_actions, length + 1, stack_dim, device=device)

        for action in range(num_actions):
            new_tapes[:, action, 0, :] = new_vecs

            if action <= length:
                # Remove action-many elements from the stack.
                new_tapes[:, action, 1:1 + length - action, :] = tapes[:, :length - action, :]
                new_tapes[:, action, 1 + length - action:, :] = 0.
            else:
                # Remove everything from the stack.
                new_tapes[:, action, 1:, :] = 0.

        new_tapes = policies * new_tapes
        new_tapes = torch.sum(new_tapes, dim=1)
        return enforce_max_depth(new_tapes, max_depth)
