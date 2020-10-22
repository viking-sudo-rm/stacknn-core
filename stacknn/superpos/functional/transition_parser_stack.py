from typing import Optional
import torch

from stacknn.superpos.functional.base import enforce_max_depth


def update_transition_parser_stack(tapes: torch.FloatTensor,
                                   policies: torch.FloatTensor,  # Distribution of shape [batch_size, 3].
                                   new_vecs: torch.FloatTensor,  # Vectors of shape [batch_size, stack_dim].
                                   max_depth: Optional[int] = None,
                                  ) -> torch.FloatTensor:
        batch_size, length, stack_dim = tapes.size()
        device = tapes.device

        if length == 0:
            left_tapes = torch.zeros(batch_size, 1, stack_dim, device=device)
            right_tapes = left_tapes
            shift_tapes = new_vecs.unsqueeze(dim=1)

        else:
            # Left-Arc operation.
            left_tapes = torch.empty(batch_size, length + 1, stack_dim, device=device)
            if length < 2:
                left_tapes[:, :-1, :] = tapes
                left_tapes[:, -1, :] = 0.
            else:
                left_tapes[:, 0, :] = tapes[:, 0]
                left_tapes[:, 1:-2, :] = tapes[:, 2:]
                left_tapes[:, -2:, :] = 0.

            # Right-Arc operation.
            right_tapes = torch.empty(batch_size, length + 1, stack_dim, device=device)
            if length < 2:
                right_tapes[:, :-1, :] = tapes
                right_tapes[:, -1, :] = 0.
            else:
                right_tapes[:, :-2, :] = tapes[:, 1:]
                right_tapes[:, -2:, :] = 0.

            # Shift operation.
            shift_tapes = torch.empty(batch_size, length + 1, stack_dim, device=device)
            shift_tapes[:, 0, :] = new_vecs
            shift_tapes[:, 1:, :] = tapes

        pol = policies.unsqueeze(-1).unsqueeze(-1)
        tapes = pol[:, 0] * left_tapes + pol[:, 1] * right_tapes + pol[:, 2] * shift_tapes
        return enforce_max_depth(tapes, max_depth)
