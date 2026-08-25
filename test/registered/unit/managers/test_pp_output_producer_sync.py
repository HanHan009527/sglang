"""Tests for the diagnostic PP output producer sync flag."""

import importlib.util
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock, call, patch

import torch

from sglang.srt.managers.scheduler_pp_mixin import (
    SchedulerPPMixin,
    _pp_can_skip_output_comm,
)

_ROOT = Path(__file__).parents[4]

_CI_REGISTER_PATH = _ROOT / "python/sglang/test/ci/ci_register.py"
_CI_SPEC = importlib.util.spec_from_file_location("ci_register", _CI_REGISTER_PATH)
ci_register = importlib.util.module_from_spec(_CI_SPEC)
assert _CI_SPEC.loader is not None
_CI_SPEC.loader.exec_module(ci_register)
register_cpu_ci = ci_register.register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_scheduler(*, debug_pp_output_producer_sync=False, is_last_rank=True):
    """Create a mock scheduler with the attributes needed by _pp_send_output_to_next_stage."""
    stream = Mock()
    stream.wait_event = Mock()
    device_module = SimpleNamespace(current_stream=Mock(return_value=stream))
    return SimpleNamespace(
        debug_pp_output_producer_sync=debug_pp_output_producer_sync,
        pp_group=SimpleNamespace(is_last_rank=is_last_rank),
        device_module=device_module,
        _pp_send_dict_to_next_stage=Mock(return_value=[Mock()]),
        ps=SimpleNamespace(gpu_id=0),
    )


def _make_target(*, forward_mode=None):
    if forward_mode is None:
        forward_mode = SimpleNamespace(is_prebuilt=Mock(return_value=False))
    return SimpleNamespace(forward_mode=forward_mode)


def _make_q_event():
    return Mock()


def _make_pp_outputs_to_send():
    return SimpleNamespace(tensors={"hidden_states": torch.zeros(1)})


# ---------------------------------------------------------------------------
# env flag existence
# ---------------------------------------------------------------------------


def test_env_flag_exists_in_environ():
    path = _ROOT / "python/sglang/srt/environ.py"
    text = path.read_text()
    assert "SGLANG_DEBUG_PP_OUTPUT_PRODUCER_SYNC" in text
    assert "EnvBool(False)" in text.split("SGLANG_DEBUG_PP_OUTPUT_PRODUCER_SYNC")[1].split("\n")[0]


# ---------------------------------------------------------------------------
# dynamic tests: flag off → no sync, wait_event → isend
# ---------------------------------------------------------------------------


def test_flag_off_no_sync_calls_wait_event_and_isend():
    scheduler = _make_scheduler(debug_pp_output_producer_sync=False)
    target = _make_target()
    q_event = _make_q_event()
    pp_outputs_to_send = _make_pp_outputs_to_send()
    comm_queue = deque()
    comm_queue.append((q_event, pp_outputs_to_send))

    with patch(
        "sglang.srt.managers.scheduler_pp_mixin._pp_can_skip_output_comm",
        return_value=False,
    ):
        SchedulerPPMixin._pp_send_output_to_next_stage(
            scheduler,
            next_first_rank_mb_id=0,
            mbs=[target],
            last_rank_comm_queue=comm_queue,
            pp_outputs=None,
        )

    q_event.synchronize.assert_not_called()
    stream = scheduler.device_module.current_stream()
    stream.wait_event.assert_called_once_with(q_event)
    scheduler._pp_send_dict_to_next_stage.assert_called_once()


# ---------------------------------------------------------------------------
# dynamic tests: flag on → sync → wait_event → isend
# ---------------------------------------------------------------------------


def test_flag_on_sync_called_before_wait_event_and_isend():
    scheduler = _make_scheduler(debug_pp_output_producer_sync=True)
    target = _make_target()
    q_event = _make_q_event()
    pp_outputs_to_send = _make_pp_outputs_to_send()
    comm_queue = deque()
    comm_queue.append((q_event, pp_outputs_to_send))

    with patch(
        "sglang.srt.managers.scheduler_pp_mixin._pp_can_skip_output_comm",
        return_value=False,
    ):
        SchedulerPPMixin._pp_send_output_to_next_stage(
            scheduler,
            next_first_rank_mb_id=0,
            mbs=[target],
            last_rank_comm_queue=comm_queue,
            pp_outputs=None,
        )

    q_event.synchronize.assert_called_once()
    stream = scheduler.device_module.current_stream()
    stream.wait_event.assert_called_once_with(q_event)
    scheduler._pp_send_dict_to_next_stage.assert_called_once()


# ---------------------------------------------------------------------------
# exception: sync raises → no wait_event, no isend, error_type is class name
# ---------------------------------------------------------------------------


def test_sync_exception_no_wait_event_no_isend_reraises():
    scheduler = _make_scheduler(debug_pp_output_producer_sync=True)
    target = _make_target()
    q_event = _make_q_event()
    q_event.synchronize = Mock(side_effect=RuntimeError("boom"))
    pp_outputs_to_send = _make_pp_outputs_to_send()
    comm_queue = deque()
    comm_queue.append((q_event, pp_outputs_to_send))

    with patch(
        "sglang.srt.managers.scheduler_pp_mixin._pp_can_skip_output_comm",
        return_value=False,
    ):
        try:
            SchedulerPPMixin._pp_send_output_to_next_stage(
                scheduler,
                next_first_rank_mb_id=0,
                mbs=[target],
                last_rank_comm_queue=comm_queue,
                pp_outputs=None,
            )
        except RuntimeError as exc:
            assert str(exc) == "boom"
        else:
            raise AssertionError("Expected RuntimeError")

    q_event.synchronize.assert_called_once()
    stream = scheduler.device_module.current_stream()
    stream.wait_event.assert_not_called()
    scheduler._pp_send_dict_to_next_stage.assert_not_called()


# ---------------------------------------------------------------------------
# phase-trace markers: before / after / error
# ---------------------------------------------------------------------------


def test_phase_trace_markers_before_and_after_sync():
    scheduler = _make_scheduler(debug_pp_output_producer_sync=True)
    target = _make_target()
    q_event = _make_q_event()
    pp_outputs_to_send = _make_pp_outputs_to_send()
    comm_queue = deque()
    comm_queue.append((q_event, pp_outputs_to_send))

    phase_tracer_mock = SimpleNamespace(enabled=True, emit=Mock())

    with (
        patch(
            "sglang.srt.managers.scheduler_pp_mixin._pp_can_skip_output_comm",
            return_value=False,
        ),
        patch(
            "sglang.srt.managers.scheduler_pp_mixin.phase_tracer",
            phase_tracer_mock,
        ),
    ):
        SchedulerPPMixin._pp_send_output_to_next_stage(
            scheduler,
            next_first_rank_mb_id=0,
            mbs=[target],
            last_rank_comm_queue=comm_queue,
            pp_outputs=None,
        )

    assert phase_tracer_mock.emit.call_count >= 2
    before_call = phase_tracer_mock.emit.call_args_list[0]
    assert before_call[0][0] == "pp_producer_sync_before"
    after_call = phase_tracer_mock.emit.call_args_list[-1]
    assert after_call[0][0] == "pp_producer_sync_after"


def test_phase_trace_error_marker_on_sync_failure():
    scheduler = _make_scheduler(debug_pp_output_producer_sync=True)
    target = _make_target()
    q_event = _make_q_event()
    q_event.synchronize = Mock(side_effect=RuntimeError("boom"))
    pp_outputs_to_send = _make_pp_outputs_to_send()
    comm_queue = deque()
    comm_queue.append((q_event, pp_outputs_to_send))

    phase_tracer_mock = SimpleNamespace(enabled=True, emit=Mock())

    with (
        patch(
            "sglang.srt.managers.scheduler_pp_mixin._pp_can_skip_output_comm",
            return_value=False,
        ),
        patch(
            "sglang.srt.managers.scheduler_pp_mixin.phase_tracer",
            phase_tracer_mock,
        ),
    ):
        try:
            SchedulerPPMixin._pp_send_output_to_next_stage(
                scheduler,
                next_first_rank_mb_id=0,
                mbs=[target],
                last_rank_comm_queue=comm_queue,
                pp_outputs=None,
            )
        except RuntimeError:
            pass

    error_calls = [
        c for c in phase_tracer_mock.emit.call_args_list
        if c[0][0] == "pp_producer_sync_error"
    ]
    assert len(error_calls) == 1
    assert error_calls[0][1]["error_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# PP0 exclusion: not is_last_rank → no sync, no wait_event
# ---------------------------------------------------------------------------


def test_pp0_not_last_rank_no_sync_no_wait_event():
    scheduler = _make_scheduler(debug_pp_output_producer_sync=True, is_last_rank=False)
    target = _make_target()
    q_event = _make_q_event()
    pp_outputs_to_send = _make_pp_outputs_to_send()
    comm_queue = deque()
    comm_queue.append((q_event, pp_outputs_to_send))

    with patch(
        "sglang.srt.managers.scheduler_pp_mixin._pp_can_skip_output_comm",
        return_value=False,
    ):
        SchedulerPPMixin._pp_send_output_to_next_stage(
            scheduler,
            next_first_rank_mb_id=0,
            mbs=[target],
            last_rank_comm_queue=comm_queue,
            pp_outputs=None,
        )

    q_event.synchronize.assert_not_called()
    stream = scheduler.device_module.current_stream()
    stream.wait_event.assert_not_called()


# ---------------------------------------------------------------------------
# Flag off: phase_tracer disabled → no emit
# ---------------------------------------------------------------------------


def test_flag_off_phase_tracer_not_called():
    scheduler = _make_scheduler(debug_pp_output_producer_sync=False)
    target = _make_target()
    q_event = _make_q_event()
    pp_outputs_to_send = _make_pp_outputs_to_send()
    comm_queue = deque()
    comm_queue.append((q_event, pp_outputs_to_send))

    phase_tracer_mock = SimpleNamespace(enabled=True, emit=Mock())

    with (
        patch(
            "sglang.srt.managers.scheduler_pp_mixin._pp_can_skip_output_comm",
            return_value=False,
        ),
        patch(
            "sglang.srt.managers.scheduler_pp_mixin.phase_tracer",
            phase_tracer_mock,
        ),
    ):
        SchedulerPPMixin._pp_send_output_to_next_stage(
            scheduler,
            next_first_rank_mb_id=0,
            mbs=[target],
            last_rank_comm_queue=comm_queue,
            pp_outputs=None,
        )

    phase_tracer_mock.emit.assert_not_called()


# ---------------------------------------------------------------------------
# env parsing
# ---------------------------------------------------------------------------


def test_env_parsing_default_false():
    import os
    from sglang.srt.environ import Envs

    assert "SGLANG_DEBUG_PP_OUTPUT_PRODUCER_SYNC" not in os.environ
    assert Envs.SGLANG_DEBUG_PP_OUTPUT_PRODUCER_SYNC.get() is False


def test_env_parsing_explicit_true():
    import os
    from sglang.srt.environ import Envs

    os.environ["SGLANG_DEBUG_PP_OUTPUT_PRODUCER_SYNC"] = "true"
    try:
        assert Envs.SGLANG_DEBUG_PP_OUTPUT_PRODUCER_SYNC.get() is True
    finally:
        os.environ.pop("SGLANG_DEBUG_PP_OUTPUT_PRODUCER_SYNC", None)
