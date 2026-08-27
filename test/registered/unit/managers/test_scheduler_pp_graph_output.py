"""CPU-only contracts for PP CUDA-graph output ownership."""

from types import SimpleNamespace

import torch

from sglang.srt.managers.scheduler_pp_mixin import (
    SchedulerPPMixin,
    _pp_snapshot_graph_output_tensors,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_proxy_send_waits_for_launch_on_schedule_stream():
    calls = []
    launch_event = object()
    payload = {"hidden_states": torch.tensor([1])}
    expected_work = [object()]
    scheduler = SimpleNamespace(
        schedule_stream=SimpleNamespace(
            wait_event=lambda event: calls.append(("wait", event))
        ),
        launch_event=launch_event,
        _pp_send_dict_to_next_stage=lambda tensor_dict, **kwargs: (
            calls.append(("send", tensor_dict, kwargs)) or expected_work
        ),
    )

    actual_work = SchedulerPPMixin._pp_send_proxy_after_launch(scheduler, payload)

    assert actual_work is expected_work
    assert calls[0] == ("wait", launch_event)
    assert calls[1][0] == "send"
    assert calls[1][1] is payload
    assert calls[1][2] == {"async_send": True, "msg_type": "proxy"}


def test_graph_output_is_detached_recursively():
    tensor = torch.tensor([1, 2, 3])
    nested = torch.tensor([4, 5])
    tuple_tensor = torch.tensor([6])
    source = {
        "token_ids": tensor,
        "nested": [nested, (tuple_tensor,)],
        "metadata": None,
    }

    snapshot = _pp_snapshot_graph_output_tensors(source, True)
    tensor.add_(10)
    nested.add_(10)
    tuple_tensor.add_(10)

    assert snapshot["token_ids"].tolist() == [1, 2, 3]
    assert snapshot["nested"][0].tolist() == [4, 5]
    assert snapshot["nested"][1][0].tolist() == [6]
    assert snapshot["token_ids"].data_ptr() != tensor.data_ptr()
    assert snapshot["nested"][0].data_ptr() != nested.data_ptr()
    assert snapshot["nested"][1][0].data_ptr() != tuple_tensor.data_ptr()


def test_eager_output_is_also_detached_for_async_relay():
    tensor = torch.tensor([1])
    source = {"token_ids": tensor}

    snapshot = _pp_snapshot_graph_output_tensors(source, False)
    tensor.add_(10)

    assert snapshot is not source
    assert snapshot["token_ids"].tolist() == [1]
    assert snapshot["token_ids"].data_ptr() != tensor.data_ptr()


def test_nested_eagle_output_is_detached_when_verify_graph_flag_is_false():
    accept_lens = torch.tensor([2, 1])
    verified_id = torch.tensor([11, 12])
    source = {
        "pp_spec_output": {
            "accept_lens": accept_lens,
            "draft_tokens": [verified_id],
        }
    }

    snapshot = _pp_snapshot_graph_output_tensors(source, False)
    accept_lens.zero_()
    verified_id.zero_()

    eagle_snapshot = snapshot["pp_spec_output"]
    assert eagle_snapshot["accept_lens"].tolist() == [2, 1]
    assert eagle_snapshot["draft_tokens"][0].tolist() == [11, 12]
    assert eagle_snapshot["accept_lens"].data_ptr() != accept_lens.data_ptr()
    assert eagle_snapshot["draft_tokens"][0].data_ptr() != verified_id.data_ptr()
