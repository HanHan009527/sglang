import torch

from sglang.srt.distributed.parallel_state import (
    TensorMetadata,
    _split_tensor_dict,
    _update_nested_dict,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_nested_tensor_dict_split_and_rebuild_round_trip():
    draft_tokens = torch.tensor([[10, 11], [20, 21]], dtype=torch.int64)
    empty = torch.empty((0, 2), dtype=torch.float32)
    original = {
        "hidden_states": torch.tensor([1.0]),
        "pp_spec_output": {
            "draft_tokens": draft_tokens,
            "optional": None,
            "deeper": {"empty_tensor": empty},
            "empty_dict": {},
        },
        "message_type": "output",
    }

    metadata, tensors = _split_tensor_dict(original)

    assert [key for key, _ in metadata] == [
        "hidden_states",
        "pp_spec_output%draft_tokens",
        "pp_spec_output%optional",
        "pp_spec_output%deeper%empty_tensor",
        "pp_spec_output%empty_dict",
        "message_type",
    ]
    assert tensors[0] is original["hidden_states"]
    assert tensors[1] is draft_tokens
    assert tensors[2] is empty
    assert isinstance(metadata[1][1], TensorMetadata)
    assert isinstance(metadata[3][1], TensorMetadata)

    rebuilt = {}
    tensor_iter = iter(tensors)
    for key, value in metadata:
        _update_nested_dict(
            rebuilt,
            key,
            next(tensor_iter) if isinstance(value, TensorMetadata) else value,
        )

    assert torch.equal(rebuilt["hidden_states"], original["hidden_states"])
    assert torch.equal(rebuilt["pp_spec_output"]["draft_tokens"], draft_tokens)
    assert torch.equal(rebuilt["pp_spec_output"]["deeper"]["empty_tensor"], empty)
    assert rebuilt["pp_spec_output"]["optional"] is None
    assert rebuilt["pp_spec_output"]["empty_dict"] == {}
    assert rebuilt["message_type"] == "output"
