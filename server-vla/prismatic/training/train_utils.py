"""
Minimal stub of prismatic.training.train_utils for SimpleVLA-RL OpenVLA-OFT inference.

Only the two mask-building functions used by modeling_prismatic.predict_action are implemented.
Their logic is derived from the inlined equivalent in the generate_action_verl method
(lines ~1166-1191 of modeling_prismatic.py).
"""

import torch
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX


def get_current_action_mask(labels: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask selecting the CURRENT action tokens in `labels`.

    The current-action positions are those in the first ACTION_DIM consecutive
    non-IGNORE positions that also have token IDs above ACTION_TOKEN_BEGIN_IDX.

    Args:
        labels: (B, seq_len) int64 tensor of token IDs; IGNORE_INDEX marks non-action slots.

    Returns:
        (B, seq_len) boolean tensor — True where current action tokens are.
    """
    newline_positions = labels != IGNORE_INDEX
    cumsum = torch.cumsum(newline_positions, dim=1)
    mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)
    action_tokens_only_mask = labels > ACTION_TOKEN_BEGIN_IDX
    return action_tokens_only_mask * mask


def get_next_actions_mask(labels: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask selecting the NEXT action chunk tokens in `labels`.

    The next-action positions are non-IGNORE positions beyond the first ACTION_DIM
    consecutive positions that also have token IDs above ACTION_TOKEN_BEGIN_IDX.

    Args:
        labels: (B, seq_len) int64 tensor of token IDs; IGNORE_INDEX marks non-action slots.

    Returns:
        (B, seq_len) boolean tensor — True where next-chunk action tokens are.
    """
    newline_positions = labels != IGNORE_INDEX
    cumsum = torch.cumsum(newline_positions, dim=1)
    mask = cumsum > ACTION_DIM
    action_tokens_only_mask = labels > ACTION_TOKEN_BEGIN_IDX
    return action_tokens_only_mask * mask
