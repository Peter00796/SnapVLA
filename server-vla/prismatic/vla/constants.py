"""
Minimal stub of prismatic.vla.constants for SimpleVLA-RL OpenVLA-OFT inference.

Values are sourced from the Haozhan72/Openvla-oft-SFT-libero10-trajall model:
  - Llama-2-7b tokenizer: vocab_size=32000, EOS=2
  - n_action_bins=256 (from config.json)
  - Action dim=7 (from dataset_statistics.json: libero_10_no_noops)
  - OFT SFT single-step: NUM_ACTIONS_CHUNK=1

NormalizationType and ACTION_PROPRIO_NORMALIZATION_TYPE:
  - Dataset statistics include q01/q99 keys, indicating BOUNDS_Q99 normalization.
"""

from enum import Enum


class NormalizationType(str, Enum):
    BOUNDS = "BOUNDS"
    BOUNDS_Q99 = "BOUNDS_Q99"


# Llama-2 uses -100 as the standard ignore index for cross-entropy
IGNORE_INDEX: int = -100

# EOS token id for Llama-2 (used as STOP token in action sequences)
STOP_INDEX: int = 2

# Action token vocabulary:
#   vocab_size=32000, n_action_bins=256
#   Action tokens occupy positions [31744, 31999] (the last 256 vocab entries)
#   Labels > ACTION_TOKEN_BEGIN_IDX selects those tokens (31744 > 31743 = True)
ACTION_TOKEN_BEGIN_IDX: int = 31743

# Action dimension (7-DoF: delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper)
ACTION_DIM: int = 7

# Number of action steps per prediction chunk (1 for base OpenVLA SFT, >1 for OFT chunk models)
NUM_ACTIONS_CHUNK: int = 1

# Normalization type — BOUNDS_Q99 uses q01/q99 percentiles (confirmed by dataset_statistics.json keys)
ACTION_PROPRIO_NORMALIZATION_TYPE: NormalizationType = NormalizationType.BOUNDS_Q99
