from __future__ import annotations

import dataclasses
import logging
from typing import Any, Tuple


logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class AsyncTransferItem:
    pool: str
    tensor_idx: int
    model_layer_id: int = dataclasses.field(default=-1, compare=False)


def collect_split_ready_transfer_items(
    forward_batch: Any,
    start_layer: int,
    end_layer: int,
) -> Tuple[AsyncTransferItem, ...]:
    attn_backend = getattr(forward_batch, "attn_backend", None)
    token_to_kv_pool = getattr(forward_batch, "token_to_kv_pool", None)
    req_to_token_pool = getattr(forward_batch, "req_to_token_pool", None)

    if attn_backend is None or token_to_kv_pool is None or req_to_token_pool is None:
        logger.warning(
            "async kv split notify skipped: missing forward batch metadata for range [%s, %s)",
            start_layer,
            end_layer,
        )
        return ()

    full_attn_layers = set(getattr(attn_backend, "full_attn_layers", ()))
    full_layer_nums = getattr(token_to_kv_pool, "full_layer_nums", len(full_attn_layers))
    use_mla = bool(getattr(token_to_kv_pool, "use_mla", False))
    full_layer_mapping = getattr(
        token_to_kv_pool, "full_attention_layer_id_mapping", {}
    )
    mamba_map = getattr(req_to_token_pool, "mamba_map", {})
    mamba_state_tensors_per_layer = int(
        getattr(attn_backend, "_mamba_state_tensors_per_layer", 0)
    )
    mamba_num_layers = int(
        getattr(attn_backend, "_mamba_num_layers", len(mamba_map) if mamba_map else 0)
    )

    transfer_items = []
    for layer_id in range(start_layer, end_layer):
        if layer_id in full_attn_layers:
            packed_id = full_layer_mapping.get(layer_id)
            if packed_id is None:
                logger.warning(
                    "async kv split notify missing full-attn mapping: model_layer=%s",
                    layer_id,
                )
                continue
            transfer_items.append(
                AsyncTransferItem(
                    pool="kv", tensor_idx=int(packed_id), model_layer_id=int(layer_id)
                )
            )
            if not use_mla:
                transfer_items.append(
                    AsyncTransferItem(
                        pool="kv",
                        tensor_idx=int(packed_id + full_layer_nums),
                        model_layer_id=int(layer_id),
                    )
                )
            continue

        mamba_layer_idx = mamba_map.get(layer_id)
        if mamba_layer_idx is None or mamba_state_tensors_per_layer <= 0:
            logger.warning(
                "async kv split notify missing linear-state mapping: model_layer=%s",
                layer_id,
            )
            continue

        for tensor_idx in range(mamba_state_tensors_per_layer):
            transfer_items.append(
                AsyncTransferItem(
                    pool="state",
                    tensor_idx=int(tensor_idx * mamba_num_layers + mamba_layer_idx),
                    model_layer_id=int(layer_id),
                )
            )

    return tuple(transfer_items)
