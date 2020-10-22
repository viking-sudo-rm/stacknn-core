from typing import Optional
import torch

from .base import enforce_max_depth


def update_kpush_stack(tapes: torch.FloatTensor,
                       policies: torch.FloatTensor,  # Distribution of shape [batch_size, num_actions].
                       new_vecs: torch.FloatTensor,  # Vectors of shape [batch_size, stack_dim].
                       num_actions: int,
                       max_depth: Optional[int],
                      ) -> torch.FloatTensor:
        batch_size, length, stack_dim = tapes.size()
        device = tapes.device

        new_vecs = new_vecs.unsqueeze(1)
        policies = policies.unsqueeze(-1).unsqueeze(-1)
        new_tapes = torch.empty(batch_size, num_actions, length + num_actions, stack_dim, device=device)

        for action in range(num_actions):
            stacked_new_vecs = new_vecs.repeat(1, action, 1)
            new_tapes[:, action, :action, :] = stacked_new_vecs

            if length > 0:
                cutoff = action + length - 1
                new_tapes[:, action, action:cutoff, :] = tapes[:, 1:, :]
                new_tapes[:, action, cutoff:, :] = 0.
            else:
                new_tapes[:, action, action:, :] = 0.

        new_tapes = policies * new_tapes
        new_tapes = torch.sum(new_tapes, dim=1)
        return enforce_max_depth(new_tapes, max_depth)
