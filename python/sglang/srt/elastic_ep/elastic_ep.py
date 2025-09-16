from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.distributed import get_expert_model_parallel_world_size


@dataclass
class ElasticEpMetadata:
    active_ranks: torch.Tensor
    last_active_ranks: torch.Tensor


_global_elastic_ep_metadata: Optional[ElasticEpMetadata] = None


def get_global_elastic_ep_metadata():
    return _global_elastic_ep_metadata


def set_global_elastic_ep_metadata(value):
    global _global_elastic_ep_metadata
    assert _global_elastic_ep_metadata is None
    _global_elastic_ep_metadata = value


def _init_global_elastic_ep_metadata(
    active_ranks: Optional[torch.Tensor] = None,
    last_active_ranks: Optional[torch.Tensor] = None,
):
    global _global_elastic_ep_metadata
    if _global_elastic_ep_metadata is not None:
        return

    if active_ranks is None:
        ep_size = get_expert_model_parallel_world_size()
        active_ranks = torch.ones(ep_size, dtype=torch.int32)

    if last_active_ranks is None:
        last_active_ranks = active_ranks.clone()

    _global_elastic_ep_metadata = ElasticEpMetadata(
        active_ranks=active_ranks, last_active_ranks=last_active_ranks
    )
