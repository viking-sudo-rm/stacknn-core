from typing import Optional
import torch

from stacknn.superpos.functional.base import enforce_max_depth


def update_minimalist_stack(tapes: torch.FloatTensor,
                            policies: torch.FloatTensor,  # Distribution of shape [batch_size, 2].
                            new_vecs: torch.FloatTensor,  # Vectors of shape [batch_size, stack_dim].
                            max_depth: Optional[int] = None,
                           ) -> torch.FloatTensor:
        batch_size, length, stack_dim = tapes.size()
        device = tapes.device

        # Push operation.
        if length == 0:
            push_tapes = new_vecs.unsqueeze(dim=1)
        else:
            push_tapes = torch.empty(batch_size, length + 1, stack_dim, device=device)
            push_tapes[:, 0, :] = new_vecs
            push_tapes[:, 1:, :] = tapes

        # Merge operation.
        merge_tapes = torch.empty(batch_size, length + 1, stack_dim, device=device)
        merge_tapes[:, 0, :] = new_vecs
        if length > 2:
            merge_tapes[:, 1:-2, :] = tapes[:, 2:]
            merge_tapes[:, -2:, :] = 0.
        else:
            merge_tapes[:, 1:, :] = 0.


        policies = policies.unsqueeze(-1).unsqueeze(-1)
        tapes = policies[:, 0] * push_tapes + policies[:, 1] * merge_tapes
        return enforce_max_depth(tapes, max_depth)
