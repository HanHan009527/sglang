"""CPU-only coverage for PP forward launch WAR ordering."""

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _SpecAlgorithm:
    @staticmethod
    def is_none():
        return False


class _RecordingQueue(deque):
    def __init__(self, order):
        super().__init__()
        self._order = order

    def append(self, item):
        self._order.append("queue_append")
        super().append(item)


@pytest.mark.parametrize("is_last_rank", [False, True])
def test_pp_launch_applies_war_barrier_before_bookkeeping(is_last_rank):
    order = []
    current_stream = object()
    event = Mock()
    event.record.side_effect = lambda stream: order.append("event_record")
    result = SimpleNamespace(can_run_cuda_graph=True)
    snapshot = object()
    batch = SimpleNamespace(
        reqs=[],
        spec_algorithm=_SpecAlgorithm(),
        copy=Mock(return_value=snapshot),
    )
    proxy_tensors = object()

    def run_batch(actual_batch, actual_proxy_tensors):
        assert actual_batch is batch
        assert actual_proxy_tensors is proxy_tensors
        order.append("run_batch")
        return result

    def apply_war_barrier():
        order.append("war_barrier")

    def set_time_batch(_reqs, field, **_kwargs):
        order.append(field)

    def make_event():
        order.append("event_create")
        return event

    def prepare_tensor_dict(actual_result, actual_batch):
        assert actual_result is result
        assert actual_batch is batch
        order.append("prepare_output")
        return {"hidden_states": object()}

    scheduler = SimpleNamespace(
        forward_stream_ctx=nullcontext(),
        forward_stream=SimpleNamespace(
            wait_stream=lambda _stream: order.append("forward_wait")
        ),
        schedule_stream=object(),
        run_batch=run_batch,
        _apply_war_barrier=apply_war_barrier,
        device_module=SimpleNamespace(
            Event=make_event,
            current_stream=lambda: current_stream,
        ),
        pp_group=SimpleNamespace(is_last_rank=is_last_rank),
        _pp_prepare_tensor_dict=prepare_tensor_dict,
    )
    metadata = [None]
    comm_queue = _RecordingQueue(order)

    with patch(
        "sglang.srt.managers.scheduler_pp_mixin.set_time_batch",
        side_effect=set_time_batch,
    ):
        actual_result, actual_event = SchedulerPPMixin._pp_launch_batch(
            scheduler,
            mb_id=0,
            cur_batch=batch,
            pp_proxy_tensors=proxy_tensors,
            mb_metadata=metadata,
            last_rank_comm_queue=comm_queue,
        )

    assert actual_result is result
    assert actual_event is event
    assert metadata[0].can_run_cuda_graph is True
    assert metadata[0].fwd_batch is snapshot
    batch.copy.assert_called_once_with()
    event.record.assert_called_once_with(current_stream)

    expected_order = [
        "forward_wait",
        "set_run_batch_cpu_start_time",
        "run_batch",
        "war_barrier",
        "set_run_batch_cpu_end_time",
    ]
    if is_last_rank:
        expected_order += ["prepare_output"]
    expected_order += ["event_create", "event_record"]
    if is_last_rank:
        expected_order += ["queue_append"]
        assert len(comm_queue) == 1
        queue_entry = comm_queue[0]
        assert queue_entry.event is event
        assert set(queue_entry.tensors.tensors) == {"hidden_states"}
        assert queue_entry.slot_id == 0
    else:
        assert not comm_queue

    assert order == expected_order
