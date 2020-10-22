from typing import Optional
import torch

from stacknn.superpos.functional.base import enforce_max_depth


def update_rewrite_stack(tapes: torch.FloatTensor,
                         policies: torch.FloatTensor,  # Distribution of shape [batch_size, 3].
                         new_vecs: torch.FloatTensor,  # Vectors of shape [batch_size, stack_dim].
                         max_depth: Optional[int] = None,
                        ) -> torch.FloatTensor:
        batch_size, length, stack_dim = tapes.size()
        device = tapes.device

        if length == 0:
            push_tapes = new_vecs.unsqueeze(dim=1)
            rewrite_tapes = torch.zeros(batch_size, 1, stack_dim, device=device)
            pop_tapes = rewrite_tapes

        else:
            # Push operation.
            push_tapes = torch.empty(batch_size, length + 1, stack_dim, device=device)
            push_tapes[:, 0, :] = new_vecs
            push_tapes[:, 1:, :] = tapes

            # Rewrite operation.
            rewrite_tapes = torch.empty(batch_size, length + 1, stack_dim, device=device)
            rewrite_tapes[:, 0, :] = new_vecs
            rewrite_tapes[:, 1:-1, :] = tapes[:, 1:, :]
            rewrite_tapes[:, -1, :] = 0.

            # Pop operation.
            pop_tapes = torch.empty(batch_size, length + 1, stack_dim, device=device)
            pop_tapes[:, :-2, :] = tapes[:, 1:]
            pop_tapes[:, -2:, :] = 0.

        policies = policies.unsqueeze(-1).unsqueeze(-1)
        tapes = policies[:, 0] * push_tapes + policies[:, 1] * rewrite_tapes + policies[:, 2] * pop_tapes
        return enforce_max_depth(tapes, max_depth)
