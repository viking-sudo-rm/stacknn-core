from typing import Optional
import torch


def enforce_max_depth(tapes: torch.FloatTensor, max_depth: Optional[int] = None) -> torch.FloatTensor:
    if max_depth is not None:
        return tapes[:, :max_depth, :]
    return tapes