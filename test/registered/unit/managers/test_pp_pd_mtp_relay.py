from collections import defaultdict, deque
from contextlib import nullcontext
from queue import Queue
from threading import Barrier, Event, Lock, Thread
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from sglang.srt.disaggregation.utils import MetadataBuffers
from sglang.srt.distributed.bootstrap import _prewarm_nccl
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler_pp_mixin import (
    PPBatchMetadata,
    SchedulerPPMixin,
    _pp_can_skip_output_comm,
    _pp_pack_control_ring_message,
    _pp_unpack_control_ring_message,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_utils import (
    get_draft_recurrent_hidden_state_spec_from_config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _SpecAlgorithm:
    @staticmethod
    def is_none():
        return False

    @staticmethod
    def is_eagle():
        return True

    @staticmethod
    def is_standalone():
        return False


class _ProxyOutputs:
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, key):
        return self.tensors[key]


def test_nccl_prewarm_initializes_distinct_tp_and_pp_groups():
    tp_group = object()
    pp_group_handle = object()
    pp_group = SimpleNamespace(
        device_group=pp_group_handle,
        ranks=[0, 8],
        rank_in_group=0,
    )
    pp_output_group_handle = object()
    pp_output_group = SimpleNamespace(
        device_group=pp_output_group_handle,
        ranks=[0, 8],
        rank_in_group=0,
    )
    warmup_tensor = object()
    recv_tensor = object()
    send_work = Mock()
    recv_work = Mock()

    with (
        patch(
            "sglang.srt.distributed.bootstrap.get_tp_group",
            return_value=SimpleNamespace(device_group=tp_group),
        ),
        patch(
            "sglang.srt.distributed.bootstrap.get_pp_group",
            return_value=pp_group,
        ),
        patch(
            "sglang.srt.distributed.bootstrap.get_pp_output_group",
            return_value=pp_output_group,
        ),
        patch(
            "sglang.srt.distributed.bootstrap.torch.zeros",
            return_value=warmup_tensor,
        ),
        patch(
            "sglang.srt.distributed.bootstrap.torch.empty_like",
            return_value=recv_tensor,
        ),
        patch(
            "sglang.srt.distributed.bootstrap.torch.cuda.current_device",
            return_value=0,
        ),
        patch("sglang.srt.distributed.bootstrap.dist.all_reduce") as all_reduce,
        patch(
            "sglang.srt.distributed.bootstrap.dist.isend", return_value=send_work
        ) as isend,
        patch(
            "sglang.srt.distributed.bootstrap.dist.irecv", return_value=recv_work
        ) as irecv,
        patch("sglang.srt.distributed.bootstrap.dist.barrier") as barrier,
        patch("sglang.srt.distributed.bootstrap.current_platform.synchronize"),
    ):
        _prewarm_nccl(tp_size=8, pp_size=2, moe_ep_size=1)

    assert all_reduce.call_args_list == [
        call(warmup_tensor, group=tp_group),
        call(warmup_tensor, group=pp_group_handle),
        call(warmup_tensor, group=pp_output_group_handle),
    ]
    assert isend.call_args_list == [
        call(warmup_tensor, dst=8, group=pp_group_handle),
        call(warmup_tensor, dst=8, group=pp_output_group_handle),
    ]
    assert irecv.call_args_list == [
        call(recv_tensor, src=8, group=pp_group_handle),
        call(recv_tensor, src=8, group=pp_output_group_handle),
    ]
    assert send_work.wait.call_count == 2
    assert recv_work.wait.call_count == 2
    assert barrier.call_args_list == [
        call(group=pp_group_handle),
        call(group=pp_group_handle),
        call(group=pp_output_group_handle),
        call(group=pp_output_group_handle),
    ]


def test_mtp_middle_chunk_skips_unused_pp_output_ring():
    batch = SimpleNamespace(
        spec_algorithm=_SpecAlgorithm(),
        forward_mode=ForwardMode.EXTEND,
        reqs=[SimpleNamespace(rid="r0")],
        contains_last_prefill_chunk=False,
        return_logprob=False,
    )

    with patch.object(
        envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM, "get", return_value=True
    ):
        assert _pp_can_skip_output_comm(batch)


def test_pp_control_ring_forwards_valid_empty_payload():
    events = []
    payload = [[], []]
    incoming = _pp_pack_control_ring_message("bootstrap", True, payload)
    control_group = object()
    scheduler = SimpleNamespace(
        pp_group=SimpleNamespace(is_last_rank=False),
        pp_disagg_control_group=control_group,
        _pp_recv_pyobj_from_prev_stage=Mock(
            side_effect=lambda group: events.append(("recv", group)) or incoming
        ),
        _pp_send_pyobj_to_next_stage=Mock(
            side_effect=lambda message, async_send, group: events.append(
                ("send", group)
            )
            or [object()]
        ),
        _pp_commit_comm_work=Mock(side_effect=lambda work: events.append("commit")),
    )
    process_payload = Mock(side_effect=lambda value: events.append("process") or value)

    result = SchedulerPPMixin._pp_run_control_ring_phase(
        scheduler,
        phase="bootstrap",
        origin_has_payload=False,
        origin_payload=None,
        process_payload=process_payload,
    )

    assert result == payload
    assert events == [
        ("recv", control_group),
        "process",
        ("send", control_group),
        "commit",
    ]
    forwarded = scheduler._pp_send_pyobj_to_next_stage.call_args.args[0]
    assert _pp_unpack_control_ring_message(forwarded, "bootstrap") == (
        True,
        payload,
    )


def test_pp_control_ring_last_stage_returns_typed_noop():
    events = []
    incoming = _pp_pack_control_ring_message("release", False, None)
    control_group = object()
    scheduler = SimpleNamespace(
        pp_group=SimpleNamespace(is_last_rank=True),
        pp_disagg_control_group=control_group,
        _pp_recv_pyobj_from_prev_stage=Mock(
            side_effect=lambda group: events.append(("recv", group)) or incoming
        ),
        _pp_send_pyobj_to_next_stage=Mock(
            side_effect=lambda message, async_send, group: events.append(
                ("send", group)
            )
            or [object()]
        ),
        _pp_commit_comm_work=Mock(side_effect=lambda work: events.append("commit")),
    )
    process_payload = Mock()

    result = SchedulerPPMixin._pp_run_control_ring_phase(
        scheduler,
        phase="release",
        origin_has_payload=False,
        origin_payload=([], []),
        process_payload=process_payload,
    )

    assert result is None
    assert events == [
        ("send", control_group),
        ("recv", control_group),
        "commit",
    ]
    process_payload.assert_not_called()
    originated = scheduler._pp_send_pyobj_to_next_stage.call_args.args[0]
    assert _pp_unpack_control_ring_message(originated, "release") == (False, None)


def test_pp_linear_payload_is_forwarded_before_following_control_phase():
    events = []
    previous_work = [object()]
    next_work = [object()]
    scheduler = SimpleNamespace(
        pp_group=SimpleNamespace(is_last_rank=False),
        _pp_commit_comm_work=Mock(
            side_effect=lambda work: events.append(("commit", work))
        ),
        _pp_send_pyobj_to_next_stage=Mock(
            side_effect=lambda payload, async_send: events.append(
                ("send", payload, async_send)
            )
            or next_work
        ),
    )

    result = SchedulerPPMixin._pp_forward_stage_payload(
        scheduler, previous_work, ["request"]
    )

    assert result is next_work
    assert events == [
        ("commit", previous_work),
        ("send", ["request"], True),
    ]


def test_pp_last_stage_consumes_linear_payload_without_forwarding():
    previous_work = [object()]
    scheduler = SimpleNamespace(
        pp_group=SimpleNamespace(is_last_rank=True),
        _pp_commit_comm_work=Mock(),
        _pp_send_pyobj_to_next_stage=Mock(),
    )

    result = SchedulerPPMixin._pp_forward_stage_payload(
        scheduler, previous_work, ["request"]
    )

    assert result == []
    scheduler._pp_commit_comm_work.assert_called_once_with(previous_work)
    scheduler._pp_send_pyobj_to_next_stage.assert_not_called()


def test_pp_proxy_exchange_is_committed_before_reusing_the_ring():
    events = []
    proxy_work = [object()]
    tensor_dict = {"hidden_states": object()}
    scheduler = SimpleNamespace(
        send_proxy_work=[],
        _pp_send_dict_to_next_stage=Mock(
            side_effect=lambda tensors, async_send, msg_type: events.append(
                ("send", tensors, async_send, msg_type)
            )
            or proxy_work
        ),
        _pp_commit_comm_work=Mock(
            side_effect=lambda work: events.append(("commit", work))
        ),
    )

    SchedulerPPMixin._pp_send_and_commit_proxy(scheduler, tensor_dict)

    assert scheduler.send_proxy_work is proxy_work
    assert events == [
        ("send", tensor_dict, True, "proxy"),
        ("commit", proxy_work),
    ]


def test_pp_proxy_and_output_use_independent_tensor_channels():
    proxy_group = SimpleNamespace(
        send_tensor_dict=Mock(return_value=[]),
        recv_tensor_dict=Mock(return_value={"__msg_type__": "proxy"}),
    )
    output_group = SimpleNamespace(
        send_tensor_dict=Mock(return_value=[]),
        recv_tensor_dict=Mock(return_value={"__msg_type__": "output"}),
    )
    all_gather_group = object()
    scheduler = SimpleNamespace(
        pp_group=proxy_group,
        pp_output_group=output_group,
        pp_output_stream_ctx=nullcontext(),
        _pp_tensor_dict_inbox=defaultdict(deque),
        require_attn_tp_allgather=False,
        attn_tp_group=all_gather_group,
    )

    proxy_tensors = {"hidden_states": object()}
    output_tensors = {"next_token_ids": object()}
    SchedulerPPMixin._pp_send_dict_to_next_stage(
        scheduler, proxy_tensors, async_send=True, msg_type="proxy"
    )
    SchedulerPPMixin._pp_send_dict_to_next_stage(
        scheduler, output_tensors, async_send=True, msg_type="output"
    )
    SchedulerPPMixin._pp_recv_typed_dict(
        scheduler, expected_kind="proxy", all_gather_group=all_gather_group
    )
    SchedulerPPMixin._pp_recv_typed_dict(
        scheduler, expected_kind="output", all_gather_group=all_gather_group
    )

    proxy_group.send_tensor_dict.assert_called_once_with(
        tensor_dict=proxy_tensors,
        all_gather_group=None,
        async_send=True,
    )
    output_group.send_tensor_dict.assert_called_once_with(
        tensor_dict=output_tensors,
        all_gather_group=None,
        async_send=True,
    )
    proxy_group.recv_tensor_dict.assert_called_once_with(
        all_gather_group=all_gather_group
    )
    output_group.recv_tensor_dict.assert_called_once_with(
        all_gather_group=all_gather_group
    )


def test_pp_disagg_output_ring_relays_fresh_payload_before_control_phase():
    events = []
    received_tensors = {"next_token_ids": object()}
    send_work = [object()]
    recorded_event = Mock()
    target = SimpleNamespace(forward_mode=SimpleNamespace(is_prebuilt=lambda: False))
    schedule_stream = object()
    output_stream = object()
    scheduler = SimpleNamespace(
        ps=SimpleNamespace(pp_rank=0),
        pp_group=SimpleNamespace(is_last_rank=False),
        copy_stream_ctx=nullcontext(),
        copy_stream=SimpleNamespace(
            wait_stream=lambda stream: events.append(("wait_stream", stream))
        ),
        schedule_stream=schedule_stream,
        pp_output_stream=output_stream,
        pp_output_stream_ctx=nullcontext(),
        device_module=SimpleNamespace(
            Event=Mock(return_value=recorded_event),
            current_stream=Mock(return_value=object()),
        ),
        _pp_recv_dict_from_prev_stage=Mock(
            side_effect=lambda: events.append("recv") or received_tensors
        ),
        _pp_prep_batch_result=Mock(
            side_effect=lambda batch, metadata, outputs: events.append("prep")
            or object()
        ),
        _pp_send_dict_to_next_stage=Mock(
            side_effect=lambda tensors, async_send, msg_type: events.append(
                ("send", tensors, async_send, msg_type)
            )
            or send_work
        ),
        _pp_send_output_to_next_stage=Mock(),
        _pp_commit_output_work=Mock(
            side_effect=lambda work: events.append(("commit", work))
        ),
    )

    with patch(
        "sglang.srt.managers.scheduler_pp_mixin._pp_can_skip_output_comm",
        return_value=False,
    ):
        outputs, _, event, work = (
            SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors(
                scheduler,
                next_first_rank_mb_id=0,
                next_mb_id=0,
                mbs=[target],
                mb_metadata=[object()],
                last_rank_comm_queue=deque(),
                pp_outputs=None,
                relay_output_immediately=True,
            )
        )

    assert outputs.tensors is received_tensors
    assert event is recorded_event
    assert work == []
    assert events[-2:] == [
        ("send", received_tensors, True, "output"),
        ("commit", send_work),
    ]
    scheduler._pp_send_output_to_next_stage.assert_not_called()


def test_pp_disagg_decode_recv_uses_forward_snapshot_after_live_slot_is_cleared():
    received_tensors = {"next_token_ids": object()}
    target = SimpleNamespace(forward_mode=SimpleNamespace(is_prebuilt=lambda: False))
    recorded_event = Mock()
    scheduler = SimpleNamespace(
        ps=SimpleNamespace(pp_rank=0),
        pp_group=SimpleNamespace(is_last_rank=False),
        copy_stream_ctx=nullcontext(),
        copy_stream=SimpleNamespace(wait_stream=Mock()),
        schedule_stream=SimpleNamespace(synchronize=Mock()),
        pp_output_stream=object(),
        pp_output_stream_ctx=nullcontext(),
        device_module=SimpleNamespace(
            Event=Mock(return_value=recorded_event),
            current_stream=Mock(return_value=object()),
        ),
        _pp_recv_dict_from_prev_stage=Mock(return_value=received_tensors),
        _pp_prep_batch_result=Mock(return_value=object()),
        _pp_send_dict_to_next_stage=Mock(return_value=[]),
        _pp_send_output_to_next_stage=Mock(return_value=[]),
        _pp_commit_comm_work=Mock(),
        _pp_commit_output_work=Mock(),
    )

    with patch(
        "sglang.srt.managers.scheduler_pp_mixin._pp_can_skip_output_comm",
        return_value=False,
    ):
        outputs, _, event, work = (
            SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors(
                scheduler,
                next_first_rank_mb_id=0,
                next_mb_id=0,
                mbs=[None],
                mb_metadata=[
                    PPBatchMetadata(can_run_cuda_graph=True, fwd_batch=target)
                ],
                last_rank_comm_queue=deque(),
                pp_outputs=None,
                relay_output_immediately=True,
                use_forward_batch_snapshot=True,
            )
        )

    assert outputs.tensors is received_tensors
    assert event is recorded_event
    assert work == []
    scheduler._pp_recv_dict_from_prev_stage.assert_called_once_with()
    scheduler._pp_prep_batch_result.assert_called_once()


def test_pp_disagg_decode_snapshot_is_single_use():
    snapshot = SimpleNamespace(
        forward_mode=SimpleNamespace(is_prebuilt=lambda: False),
        reqs=[],
    )
    live_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_prebuilt=lambda: False),
        reqs=[],
    )
    metadata = PPBatchMetadata(can_run_cuda_graph=True, fwd_batch=snapshot)
    scheduler = SimpleNamespace(
        mbs=[live_batch, None],
        mb_metadata=[metadata, None],
        last_mbs=[None, None],
        _pp_process_batch_result=Mock(),
    )
    result = object()
    d2h_event = Mock()

    SchedulerPPMixin._pp_process_relayed_batch_result(
        scheduler,
        0,
        result,
        d2h_event,
        use_forward_batch_snapshot=True,
    )

    scheduler._pp_process_batch_result.assert_called_once_with(snapshot, result)
    assert scheduler.last_mbs[0] is live_batch
    assert scheduler.mb_metadata[0] is None


def test_pp_disagg_decode_forward_snapshot_preserves_seq_lens():
    seq_lens = torch.tensor([17, 23], dtype=torch.int64)
    batch = ScheduleBatch(reqs=[], seq_lens=seq_lens)

    snapshot = batch.copy()

    # EAGLE PP result processing advances this tensor by accept_lens.  A
    # forward snapshot without seq_lens fails there with ``None + Tensor``.
    assert snapshot.seq_lens is seq_lens
    assert torch.equal(snapshot.seq_lens + torch.tensor([2, 3]), torch.tensor([19, 26]))


def test_pp_disagg_decode_processes_snapshot_after_live_slot_is_cleared():
    snapshot = SimpleNamespace(
        forward_mode=SimpleNamespace(is_prebuilt=lambda: False),
        reqs=[],
    )
    scheduler = SimpleNamespace(
        mbs=[None],
        mb_metadata=[PPBatchMetadata(can_run_cuda_graph=True, fwd_batch=snapshot)],
        last_mbs=[object()],
        _pp_process_batch_result=Mock(),
    )
    result = object()
    d2h_event = Mock()

    SchedulerPPMixin._pp_process_relayed_batch_result(
        scheduler,
        0,
        result,
        d2h_event,
        use_forward_batch_snapshot=True,
    )

    d2h_event.synchronize.assert_called_once_with()
    scheduler._pp_process_batch_result.assert_called_once_with(snapshot, result)
    assert scheduler.last_mbs[0] is None
    assert scheduler.mb_metadata[0] is None


def test_pp_disagg_output_ring_last_stage_starts_relay_chain():
    events = []
    send_work = [object()]
    target = SimpleNamespace(forward_mode=SimpleNamespace(is_prebuilt=lambda: False))
    recorded_event = Mock()
    scheduler = SimpleNamespace(
        ps=SimpleNamespace(pp_rank=1),
        pp_group=SimpleNamespace(is_last_rank=True),
        copy_stream_ctx=nullcontext(),
        copy_stream=SimpleNamespace(wait_stream=Mock()),
        schedule_stream=object(),
        pp_output_stream=object(),
        pp_output_stream_ctx=nullcontext(),
        device_module=SimpleNamespace(
            Event=Mock(return_value=recorded_event),
            current_stream=Mock(return_value=object()),
        ),
        _pp_send_output_to_next_stage=Mock(
            side_effect=lambda *args: events.append("send") or send_work
        ),
        _pp_recv_dict_from_prev_stage=Mock(
            side_effect=lambda: events.append("recv") or {"next_token_ids": object()}
        ),
        _pp_prep_batch_result=Mock(return_value=object()),
        _pp_commit_comm_work=Mock(
            side_effect=lambda work: events.append(("commit", work))
        ),
        _pp_commit_output_work=Mock(),
    )

    with patch(
        "sglang.srt.managers.scheduler_pp_mixin._pp_can_skip_output_comm",
        return_value=False,
    ):
        _, _, _, work = SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors(
            scheduler,
            next_first_rank_mb_id=0,
            next_mb_id=0,
            mbs=[target],
            mb_metadata=[object()],
            last_rank_comm_queue=deque(),
            pp_outputs=None,
            relay_output_immediately=True,
        )

    assert work is send_work
    assert events == ["recv", "send"]
    scheduler._pp_commit_comm_work.assert_not_called()


def _run_closed_output_ring_rank(pp_rank, *, send_work, target):
    events = []
    scheduler = SimpleNamespace(
        ps=SimpleNamespace(pp_rank=pp_rank),
        pp_group=SimpleNamespace(is_last_rank=pp_rank == 1),
        copy_stream_ctx=nullcontext(),
        copy_stream=SimpleNamespace(wait_stream=Mock()),
        pp_output_stream=object(),
        device_module=SimpleNamespace(
            Event=Mock(return_value=Mock()), current_stream=Mock(return_value=object())
        ),
        _pp_send_output_to_next_stage=Mock(
            side_effect=lambda output_mb_id, *_args, **_kwargs: events.append(
                ("origin_send", output_mb_id)
            )
            or send_work
        ),
        _pp_recv_dict_from_prev_stage=Mock(
            side_effect=lambda: events.append("recv") or {"next_token_ids": object()}
        ),
        _pp_send_dict_to_next_stage=Mock(
            side_effect=lambda *_args, **_kwargs: events.append("reverse_send")
            or send_work
        ),
        _pp_prep_batch_result=Mock(return_value=object()),
        _pp_commit_output_work=Mock(
            side_effect=lambda work: events.append(("commit", work))
        ),
    )

    with patch(
        "sglang.srt.managers.scheduler_pp_mixin._pp_can_skip_output_comm",
        return_value=False,
    ):
        result = SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors(
            scheduler,
            next_first_rank_mb_id=1,
            next_mb_id=0,
            mbs=[target, None],
            mb_metadata=[SimpleNamespace(fwd_batch=None), None],
            last_rank_comm_queue=deque(),
            pp_outputs=None,
            relay_output_immediately=True,
            use_forward_batch_snapshot=True,
            close_output_ring=True,
        )

    return events, result


def test_pp_disagg_output_ring_closes_with_paired_rank_order():
    target = SimpleNamespace(forward_mode=SimpleNamespace(is_prebuilt=lambda: False))
    send_work = [object()]

    pp0_events, pp0_result = _run_closed_output_ring_rank(
        0, send_work=send_work, target=target
    )
    pp1_events, pp1_result = _run_closed_output_ring_rank(
        1, send_work=send_work, target=target
    )

    assert pp0_events == ["recv", "reverse_send", ("commit", send_work)]
    assert pp1_events == [("origin_send", 0), "recv", ("commit", send_work)]
    assert pp0_result[3] == pp1_result[3] == []


def test_pp_disagg_output_ring_two_rank_multislot_has_no_cross_slot_work():
    target = SimpleNamespace(forward_mode=SimpleNamespace(is_prebuilt=lambda: False))
    forward = Queue()
    reverse = Queue()
    slot_barrier = Barrier(2)
    events = []
    events_lock = Lock()
    results = {0: [], 1: []}

    def record(pp_rank, slot, event):
        with events_lock:
            events.append((pp_rank, slot, event))

    def send(channel, payload, pp_rank, slot, event):
        done = Event()
        record(pp_rank, slot, event)
        channel.put((payload, done))
        return [SimpleNamespace(work=SimpleNamespace(wait=done.wait))]

    def recv(channel, pp_rank, slot):
        payload, done = channel.get(timeout=2)
        record(pp_rank, slot, "recv")
        done.set()
        return payload

    def run_rank(pp_rank):
        current_slot = [0]
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(pp_rank=pp_rank),
            pp_group=SimpleNamespace(is_last_rank=pp_rank == 1),
            copy_stream_ctx=nullcontext(),
            copy_stream=SimpleNamespace(wait_stream=Mock()),
            pp_output_stream=object(),
            device_module=SimpleNamespace(
                Event=Mock(return_value=Mock()),
                current_stream=Mock(return_value=object()),
            ),
            _pp_send_output_to_next_stage=lambda *_args, **_kwargs: send(
                forward,
                {"next_token_ids": object()},
                pp_rank,
                current_slot[0],
                "origin_send",
            ),
            _pp_recv_dict_from_prev_stage=lambda: recv(
                forward if pp_rank == 0 else reverse, pp_rank, current_slot[0]
            ),
            _pp_send_dict_to_next_stage=lambda payload, **_kwargs: send(
                reverse, payload, pp_rank, current_slot[0], "reverse_send"
            ),
            _pp_prep_batch_result=Mock(return_value=object()),
        )

        def commit(work):
            for item in work:
                assert item.work.wait(timeout=2)
            work.clear()
            record(pp_rank, current_slot[0], "pg8_complete")

        scheduler._pp_commit_output_work = commit
        for slot in range(2):
            current_slot[0] = slot
            # Models the event-loop gate: both ranks finish PG7 before either
            # rank is allowed to issue this slot's PG8 ring.
            record(pp_rank, slot, "pg7_complete")
            slot_barrier.wait(timeout=2)
            _, _, _, work = (
                SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors(
                    scheduler,
                    next_first_rank_mb_id=1,
                    next_mb_id=0,
                    mbs=[target, target],
                    mb_metadata=[
                        SimpleNamespace(fwd_batch=None),
                        SimpleNamespace(fwd_batch=None),
                    ],
                    last_rank_comm_queue=deque(),
                    pp_outputs=None,
                    relay_output_immediately=True,
                    use_forward_batch_snapshot=True,
                    close_output_ring=True,
                )
            )
            results[pp_rank].append(work)
            slot_barrier.wait(timeout=2)

    with patch(
        "sglang.srt.managers.scheduler_pp_mixin._pp_can_skip_output_comm",
        return_value=False,
    ):
        threads = [Thread(target=run_rank, args=(rank,)) for rank in (0, 1)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()

    assert results == {0: [[], []], 1: [[], []]}
    for slot in range(2):
        slot_events = [event for _, event_slot, event in events if event_slot == slot]
        assert slot_events.count("pg7_complete") == 2
        first_pg8 = min(
            index
            for index, (_, event_slot, event) in enumerate(events)
            if event_slot == slot and event not in {"pg7_complete"}
        )
        last_pg7 = max(
            index
            for index, (_, event_slot, event) in enumerate(events)
            if event_slot == slot and event == "pg7_complete"
        )
        assert last_pg7 < first_pg8
        assert slot_events.count("pg8_complete") == 2

    last_slot0_pg8 = max(
        index
        for index, (_, slot, event) in enumerate(events)
        if slot == 0 and event == "pg8_complete"
    )
    first_slot1_pg7 = min(
        index
        for index, (_, slot, event) in enumerate(events)
        if slot == 1 and event == "pg7_complete"
    )
    assert last_slot0_pg8 < first_slot1_pg7


def test_pp_disagg_output_ring_initial_empty_slot_is_symmetric_and_non_consuming():
    origin_target = object()
    for pp_rank in (0, 1):
        events = []
        queue = deque([object()])
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(pp_rank=pp_rank),
            pp_group=SimpleNamespace(is_last_rank=pp_rank == 1),
            _pp_send_output_to_next_stage=Mock(
                side_effect=lambda output_mb_id, mbs, *_args, **_kwargs: (
                    events.append(("empty", output_mb_id)) or []
                    if mbs[output_mb_id] is None
                    else events.append(("send", output_mb_id)) or [object()]
                )
            ),
            _pp_recv_dict_from_prev_stage=Mock(
                side_effect=lambda: events.append("recv") or {}
            ),
            _pp_send_dict_to_next_stage=Mock(),
            _pp_commit_output_work=Mock(
                side_effect=lambda work: events.append(("commit", work))
            ),
        )

        _, _, _, work = SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors(
            scheduler,
            next_first_rank_mb_id=0,
            next_mb_id=1,
            mbs=[origin_target, None],
            mb_metadata=[object(), None],
            last_rank_comm_queue=queue,
            pp_outputs=None,
            relay_output_immediately=True,
            use_forward_batch_snapshot=True,
            close_output_ring=True,
        )

        assert events == (
            [("commit", [])] if pp_rank == 0 else [("empty", 1), ("commit", [])]
        )
        assert len(queue) == 1
        assert work == []
        scheduler._pp_recv_dict_from_prev_stage.assert_not_called()
        scheduler._pp_send_dict_to_next_stage.assert_not_called()


def test_pp_disagg_output_ring_skip_slot_is_symmetric_and_has_no_wire_work():
    target = SimpleNamespace(forward_mode=SimpleNamespace(is_prebuilt=lambda: False))
    metadata = SimpleNamespace(fwd_batch=None)
    skip_result = (None, object(), object())

    for pp_rank in (0, 1):
        queue = deque([(Mock(), object())])
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(pp_rank=pp_rank),
            pp_group=SimpleNamespace(is_last_rank=pp_rank == 1),
            _pp_recv_dict_from_prev_stage=Mock(),
            _pp_send_dict_to_next_stage=Mock(),
            _pp_make_skip_output_result=Mock(return_value=skip_result),
            _pp_commit_output_work=Mock(),
        )
        scheduler._pp_send_output_to_next_stage = lambda *args, **kwargs: (
            SchedulerPPMixin._pp_send_output_to_next_stage(scheduler, *args, **kwargs)
        )

        with patch(
            "sglang.srt.managers.scheduler_pp_mixin._pp_can_skip_output_comm",
            return_value=True,
        ):
            next_outputs, batch_result, d2h_event, work = (
                SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors(
                    scheduler,
                    next_first_rank_mb_id=1,
                    next_mb_id=0,
                    mbs=[target, None],
                    mb_metadata=[metadata, None],
                    last_rank_comm_queue=queue,
                    pp_outputs=None,
                    relay_output_immediately=True,
                    use_forward_batch_snapshot=True,
                    close_output_ring=True,
                )
            )

        assert (next_outputs, batch_result, d2h_event) == skip_result
        assert work == []
        scheduler._pp_recv_dict_from_prev_stage.assert_not_called()
        scheduler._pp_send_dict_to_next_stage.assert_not_called()
        scheduler._pp_commit_output_work.assert_called_once_with([])
        assert len(queue) == (0 if pp_rank == 1 else 1)


def test_pp_disagg_output_origin_send_survives_empty_return_slot():
    send_work = [object()]
    origin_target = object()
    scheduler = SimpleNamespace(
        ps=SimpleNamespace(pp_rank=1),
        pp_group=SimpleNamespace(is_last_rank=True),
        _pp_send_output_to_next_stage=Mock(return_value=send_work),
        _pp_recv_dict_from_prev_stage=Mock(),
        _pp_commit_comm_work=Mock(),
    )

    _, _, _, work = SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors(
        scheduler,
        next_first_rank_mb_id=0,
        next_mb_id=1,
        mbs=[origin_target, None],
        mb_metadata=[object(), None],
        last_rank_comm_queue=deque(),
        pp_outputs=None,
        relay_output_immediately=True,
    )

    assert work is send_work
    scheduler._pp_send_output_to_next_stage.assert_called_once()
    scheduler._pp_recv_dict_from_prev_stage.assert_not_called()
    scheduler._pp_commit_comm_work.assert_not_called()


def test_pp_output_ring_uses_rank_parity_for_send_recv_order():
    target = SimpleNamespace(forward_mode=SimpleNamespace(is_prebuilt=lambda: False))

    for pp_rank, expected_order in ((0, ["send", "recv"]), (1, ["recv", "send"])):
        events = []
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(pp_rank=pp_rank),
            pp_group=SimpleNamespace(is_last_rank=pp_rank == 1),
            copy_stream_ctx=nullcontext(),
            copy_stream=SimpleNamespace(wait_stream=Mock()),
            schedule_stream=object(),
            pp_output_stream=object(),
            pp_output_stream_ctx=nullcontext(),
            device_module=SimpleNamespace(
                Event=Mock(return_value=Mock()),
                current_stream=Mock(return_value=object()),
            ),
            _pp_send_output_to_next_stage=Mock(
                side_effect=lambda *_args: events.append("send") or []
            ),
            _pp_recv_dict_from_prev_stage=Mock(
                side_effect=lambda: events.append("recv")
                or {"next_token_ids": object()}
            ),
            _pp_prep_batch_result=Mock(return_value=object()),
        )

        with patch(
            "sglang.srt.managers.scheduler_pp_mixin._pp_can_skip_output_comm",
            return_value=False,
        ):
            SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors(
                scheduler,
                next_first_rank_mb_id=0,
                next_mb_id=0,
                mbs=[target],
                mb_metadata=[object()],
                last_rank_comm_queue=deque(),
                pp_outputs=None,
                relay_output_immediately=False,
            )

        assert events == expected_order


def test_pp_output_channel_uses_dedicated_stream_for_send_recv_and_commit():
    events = []

    class RecordingContext:
        def __enter__(self):
            events.append("output_stream_enter")

        def __exit__(self, exc_type, exc, tb):
            events.append("output_stream_exit")

    proxy_group = SimpleNamespace(
        send_tensor_dict=Mock(
            side_effect=lambda **kwargs: events.append("proxy_send") or []
        ),
        recv_tensor_dict=Mock(),
    )
    output_group = SimpleNamespace(
        send_tensor_dict=Mock(
            side_effect=lambda **kwargs: events.append("output_send") or []
        ),
        recv_tensor_dict=Mock(
            side_effect=lambda **kwargs: events.append("output_recv")
            or {"__msg_type__": "output"}
        ),
    )
    scheduler = SimpleNamespace(
        pp_group=proxy_group,
        pp_output_group=output_group,
        pp_output_stream_ctx=RecordingContext(),
        _pp_tensor_dict_inbox=defaultdict(deque),
        require_attn_tp_allgather=False,
        attn_tp_group=object(),
        _pp_commit_comm_work=Mock(
            side_effect=lambda work: events.append(("output_commit", work))
        ),
    )

    SchedulerPPMixin._pp_send_dict_to_next_stage(
        scheduler, {"x": object()}, msg_type="proxy"
    )
    SchedulerPPMixin._pp_send_dict_to_next_stage(
        scheduler, {"y": object()}, msg_type="output"
    )
    SchedulerPPMixin._pp_recv_typed_dict(scheduler, expected_kind="output")
    work = [object()]
    SchedulerPPMixin._pp_commit_output_work(scheduler, work)

    assert events == [
        "proxy_send",
        "output_stream_enter",
        "output_send",
        "output_stream_exit",
        "output_stream_enter",
        "output_recv",
        "output_stream_exit",
        "output_stream_enter",
        ("output_commit", work),
        "output_stream_exit",
    ]


def test_pp_prefill_rebuilds_one_authoritative_draft_input():
    topk_p = torch.randn(2, 1)
    topk_index = torch.tensor([[3], [7]], dtype=torch.int64)
    hidden_states = torch.randn(2, 8)
    bonus_tokens = torch.tensor([11, 13], dtype=torch.int64)
    dsa_topk_indices = torch.tensor([[2, 4], [6, 8]], dtype=torch.int32)
    pp_outputs = _ProxyOutputs(
        {
            "next_token_ids": bonus_tokens,
            "spec_prefill_topk_p": topk_p,
            "spec_prefill_topk_index": topk_index,
            "spec_prefill_hidden_states": hidden_states,
            "spec_prefill_dsa_topk_indices": dsa_topk_indices,
        }
    )
    batch = SimpleNamespace(
        spec_algorithm=_SpecAlgorithm(),
        reqs=[SimpleNamespace(rid="r0"), SimpleNamespace(rid="r1")],
        req_pool_indices=torch.tensor([0, 1]),
        forward_mode=SimpleNamespace(is_extend=Mock(return_value=True)),
        return_logprob=False,
        input_ids=bonus_tokens.clone(),
        spec_info=None,
    )
    scheduler = SimpleNamespace(
        spec_algorithm=_SpecAlgorithm(),
        server_args=SimpleNamespace(speculative_num_draft_tokens=5),
        future_map=SimpleNamespace(stash=Mock()),
        device_module=SimpleNamespace(Event=Mock(return_value=Mock())),
        _pp_spec_store_bonus=Mock(),
    )

    with patch.object(
        GenerationBatchResult, "copy_to_cpu", autospec=True
    ) as copy_to_cpu:
        result = SchedulerPPMixin._pp_prep_batch_result(
            scheduler,
            batch,
            PPBatchMetadata(can_run_cuda_graph=True, fwd_batch=None),
            pp_outputs,
        )

    assert batch.spec_info is result.next_draft_input
    assert result.copy_done is None
    copy_to_cpu.assert_not_called()
    assert result.next_draft_input.topk_p is topk_p
    assert result.next_draft_input.topk_index is topk_index
    assert result.next_draft_input.hidden_states is hidden_states
    assert result.next_draft_input.bonus_tokens is bonus_tokens
    assert result.next_draft_input.dsa_topk_indices is dsa_topk_indices
    assert batch.input_ids is None
    scheduler.future_map.stash.assert_called_once()


def test_pp_spec_decode_copies_cpu_bound_result_before_processing():
    bonus_tokens = torch.tensor([11, 13], dtype=torch.int64)
    pp_outputs = _ProxyOutputs(
        {
            "next_token_ids": bonus_tokens,
            "pp_spec_output": {
                "draft_tokens": torch.tensor([[11, 2, 3, 4], [13, 5, 6, 7]]),
                "bonus_tokens": bonus_tokens,
                "top_scores_index": torch.tensor([[0, 1, 2], [0, 1, 2]]),
                "parent_list": torch.tensor([[-1, 0, 1], [-1, 0, 1]]),
                "accept_lens": torch.tensor([3, 2]),
                "accept_index": None,
            },
        }
    )
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(rid="r0"), SimpleNamespace(rid="r1")],
        req_pool_indices=torch.tensor([0, 1]),
        forward_mode=SimpleNamespace(is_extend=Mock(return_value=False)),
        return_logprob=False,
        input_ids=bonus_tokens.clone(),
        spec_info=None,
    )
    copy_done = Mock()
    scheduler = SimpleNamespace(
        spec_algorithm=_SpecAlgorithm(),
        server_args=SimpleNamespace(speculative_num_draft_tokens=4),
        future_map=SimpleNamespace(stash=Mock()),
        device_module=SimpleNamespace(Event=Mock(return_value=copy_done)),
    )

    with patch.object(
        GenerationBatchResult, "copy_to_cpu", autospec=True
    ) as copy_to_cpu:
        result = SchedulerPPMixin._pp_prep_batch_result(
            scheduler,
            batch,
            PPBatchMetadata(can_run_cuda_graph=True, fwd_batch=None),
            pp_outputs,
        )

    assert result.copy_done is copy_done
    assert result.accept_lens.tolist() == [3, 2]
    assert result.speculative_num_draft_tokens == 4
    copy_to_cpu.assert_called_once_with(
        result,
        return_logprob=False,
        return_hidden_states=False,
    )


def test_spec_only_aux_indices_follow_optional_sampling_mask_layout():
    for sampling_mask_tokens, expected in (
        (0, [6, 7, 8, 9]),
        (32, [9, 10, 11, 12]),
    ):
        buffers = MetadataBuffers(
            size=2,
            hidden_size=8,
            hidden_states_dtype=torch.float32,
            max_sampling_mask_tokens=sampling_mask_tokens,
            output_dsa_topk_indices_dim=4,
        )
        ptrs, _, _ = buffers.get_buf_infos()
        indices = buffers.get_spec_only_aux_indices()
        assert indices == expected
        assert ptrs[indices[0]] == buffers.output_topk_p.data_ptr()
        assert ptrs[indices[1]] == buffers.output_topk_index.data_ptr()
        assert ptrs[indices[2]] == buffers.output_hidden_states.data_ptr()
        assert ptrs[indices[3]] == buffers.output_dsa_topk_indices.data_ptr()

    buffers_without_dsa = MetadataBuffers(
        size=2,
        hidden_size=8,
        hidden_states_dtype=torch.float32,
        max_sampling_mask_tokens=0,
        output_dsa_topk_indices_dim=0,
    )
    assert buffers_without_dsa.get_spec_only_aux_indices() == [6, 7, 8]


def test_draft_hidden_state_wire_schema_does_not_require_a_local_runner():
    config = SimpleNamespace(spec_hidden_size=6144, dtype=torch.bfloat16)

    hidden_size, dtype = get_draft_recurrent_hidden_state_spec_from_config(
        config, _SpecAlgorithm()
    )

    assert hidden_size == 6144
    assert dtype is torch.bfloat16
