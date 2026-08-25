import importlib.util
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

_ROOT = Path(__file__).parents[4]
_TEST_MODULES = {}
for package_name, package_path in (
    ("sglang", _ROOT / "python/sglang"),
    ("sglang.srt", _ROOT / "python/sglang/srt"),
    ("sglang.srt.debug_utils", _ROOT / "python/sglang/srt/debug_utils"),
):
    package = ModuleType(package_name)
    package.__path__ = [str(package_path)]
    _TEST_MODULES[package_name] = package
_TEST_MODULES["sglang.srt.debug_utils.cuda_coredump"] = ModuleType(
    "sglang.srt.debug_utils.cuda_coredump"
)
_ENVIRON_PATH = _ROOT / "python/sglang/srt/environ.py"
_ENVIRON_SPEC = importlib.util.spec_from_file_location(
    "sglang.srt.environ", _ENVIRON_PATH
)
environ = importlib.util.module_from_spec(_ENVIRON_SPEC)
assert _ENVIRON_SPEC.loader is not None
with patch.dict(sys.modules, _TEST_MODULES):
    _ENVIRON_SPEC.loader.exec_module(environ)
    sys.modules["sglang.srt.environ"] = environ
    _MODULE_PATH = _ROOT / "python/sglang/srt/managers/phase_trace.py"
    _SPEC = importlib.util.spec_from_file_location(
        "phase_trace_under_test", _MODULE_PATH
    )
    phase_trace = importlib.util.module_from_spec(_SPEC)
    assert _SPEC.loader is not None
    _SPEC.loader.exec_module(phase_trace)
PhaseTracer = phase_trace.PhaseTracer
batch_phase_fields = phase_trace.batch_phase_fields
describe_process_group = phase_trace.describe_process_group
format_phase_trace = phase_trace.format_phase_trace
format_phase_trace_snapshot = phase_trace.format_phase_trace_snapshot


_CI_REGISTER_PATH = _ROOT / "python/sglang/test/ci/ci_register.py"
_CI_SPEC = importlib.util.spec_from_file_location("ci_register", _CI_REGISTER_PATH)
ci_register = importlib.util.module_from_spec(_CI_SPEC)
assert _CI_SPEC.loader is not None
_CI_SPEC.loader.exec_module(ci_register)
register_cpu_ci = ci_register.register_cpu_ci


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Mode:
    def __init__(self, name: str, idle: bool):
        self.name = name
        self._idle = idle

    def is_idle(self):
        return self._idle


def _payload(line: str) -> dict:
    return json.loads(line.split(" ", 1)[1])


def test_format_phase_trace_covers_pp0_pp1_and_avoids_object_introspection():
    opaque = object()
    for pp_rank in (0, 1):
        payload = _payload(
            format_phase_trace(
                "pp_recv_before",
                7 + pp_rank,
                {
                    "global_rank": pp_rank * 8,
                    "pp_rank": pp_rank,
                    "dp_rank": 0,
                    "microbatch_id": 3,
                    "opaque": opaque,
                },
            )
        )
        assert payload["event"] == "pp_recv_before"
        assert payload["pp_rank"] == pp_rank
        assert payload["opaque"] == "<object>"


def test_batch_phase_fields_handles_active_idle_and_missing_batches():
    active = SimpleNamespace(
        forward_mode=_Mode("DECODE", False),
        reqs=[1, 2],
        can_run_dp_cuda_graph=True,
        can_run_dp_draft_cuda_graph=False,
        global_num_tokens=[2, 0],
    )
    idle = SimpleNamespace(forward_mode=_Mode("IDLE", True), reqs=[])

    assert batch_phase_fields(active) | {"batch_id": None} == {
        "active": True,
        "batch_id": None,
        "batch_size": 2,
        "forward_mode": "DECODE",
        "batch_global_decode_graph_vote": True,
        "batch_global_draft_graph_vote": False,
        "batch_global_tokens": [2, 0],
        "idle": False,
    }
    assert batch_phase_fields(idle)["idle"] is True
    assert batch_phase_fields(None) == {
        "active": False,
        "batch_id": None,
        "batch_size": 0,
        "forward_mode": None,
        "batch_global_decode_graph_vote": None,
        "batch_global_draft_graph_vote": None,
        "batch_global_tokens": None,
        "idle": True,
    }


def test_describe_process_group_supports_gloo_nccl_and_fallback():
    phase_trace.clear_process_group_metadata_cache()
    for backend in ("gloo", "nccl"):
        group = object()
        dist = SimpleNamespace(
            get_backend=Mock(return_value=backend),
            get_world_size=Mock(return_value=2),
            get_rank=Mock(return_value=1),
            get_process_group_ranks=Mock(return_value=[0, 8]),
        )
        assert describe_process_group(group, dist=dist) == {
            "pg_backend": backend,
            "pg_members": [0, 8],
            "pg_rank": 1,
            "pg_size": 2,
        }
        assert describe_process_group(group, dist=dist) == {
            "pg_backend": backend,
            "pg_members": [0, 8],
            "pg_rank": 1,
            "pg_size": 2,
        }
        dist.get_backend.assert_called_once_with(group)
        dist.get_world_size.assert_called_once_with(group)
        dist.get_rank.assert_called_once_with(group)
        dist.get_process_group_ranks.assert_called_once_with(group)

    failure_group = object()
    failure = Mock(side_effect=RuntimeError)
    dist = SimpleNamespace(
        get_backend=failure,
        get_world_size=failure,
        get_rank=failure,
        get_process_group_ranks=failure,
    )
    assert describe_process_group(
        failure_group, dist=dist, known_members=[8, 9], known_rank=0
    ) == {
        "pg_backend": "unknown",
        "pg_members": [8, 9],
        "pg_rank": 0,
        "pg_size": 2,
    }


def test_phase_tracer_disabled_path_does_not_format_or_log():
    log = Mock()
    disabled = PhaseTracer(enabled=False, max_events=2, every_n=1, ring_size=2, log=log)
    with patch.object(phase_trace, "_make_phase_trace_record") as formatter:
        assert disabled.emit("disabled", tensor=object()) is False
    formatter.assert_not_called()
    log.info.assert_not_called()
    log.warning.assert_not_called()
    assert disabled.snapshot() == {
        "enabled": False,
        "last_marker": None,
        "ring_size": 2,
        "ring_tail": [],
    }
    dist = SimpleNamespace(
        get_backend=Mock(),
        get_world_size=Mock(),
        get_rank=Mock(),
        get_process_group_ranks=Mock(),
    )
    # Call sites guard PG metadata collection with this same startup-cached
    # boolean, so the default-off hot path never reaches introspection.
    if disabled.enabled:
        describe_process_group(object(), dist=dist)
    for method in (
        dist.get_backend,
        dist.get_world_size,
        dist.get_rank,
        dist.get_process_group_ranks,
    ):
        method.assert_not_called()


def test_phase_tracer_ring_last_marker_sampling_and_log_bound():
    log = Mock()
    enabled = PhaseTracer(enabled=True, max_events=2, every_n=2, ring_size=3, log=log)
    assert enabled.emit("skipped") is False
    assert enabled.emit("first", local_tokens=1) is True
    assert enabled.emit("skipped") is False
    assert enabled.emit("second", global_tokens=[1, 0]) is True
    assert enabled.emit("skipped") is False
    assert enabled.emit("limited") is False
    assert log.info.call_count == 2
    log.warning.assert_called_once()
    snapshot = enabled.snapshot(tail=32)
    assert snapshot["ring_size"] == 3
    assert [record["event"] for record in snapshot["ring_tail"]] == [
        "second",
        "skipped",
        "limited",
    ]
    assert enabled.last_marker()["event"] == "limited"


def test_process_group_metadata_is_cached_by_group_identity():
    phase_trace.clear_process_group_metadata_cache()
    group = object()
    dist = SimpleNamespace(
        get_backend=Mock(return_value="nccl"),
        get_world_size=Mock(return_value=2),
        get_rank=Mock(return_value=0),
        get_process_group_ranks=Mock(return_value=[0, 8]),
    )

    first = describe_process_group(group, dist=dist)
    second = describe_process_group(group, dist=dist)

    assert first == second
    for method in (
        dist.get_backend,
        dist.get_world_size,
        dist.get_rank,
        dist.get_process_group_ranks,
    ):
        method.assert_called_once()


def test_logger_failure_does_not_escape_and_ring_remains_visible():
    log = Mock()
    log.info.side_effect = RuntimeError("handler failed")
    tracer = PhaseTracer(enabled=True, max_events=4, every_n=1, ring_size=4, log=log)

    assert tracer.emit("before_failure", local_tokens=3) is False
    assert tracer.last_marker()["event"] == "before_failure"
    assert tracer.snapshot()["ring_tail"][0]["local_tokens"] == 3


def test_hostile_fields_and_recursive_containers_never_escape():
    class Hostile:
        @property
        def value(self):
            raise RuntimeError("hostile property")

    tracer = PhaseTracer(enabled=True, max_events=4, every_n=1, ring_size=4, log=Mock())
    assert tracer.emit("hostile", collect=lambda: {"value": Hostile().value}) is False
    assert tracer.last_marker() is None

    recursive = []
    recursive.append(recursive)
    assert tracer.emit("recursive", value=recursive) is True
    assert tracer.last_marker()["value"] == ["<list:cycle>"]


def test_log_limit_stops_logging_and_pg_introspection_but_keeps_ring():
    log = Mock()
    tracer = PhaseTracer(enabled=True, max_events=1, every_n=1, ring_size=3, log=log)
    pg_probe = Mock(return_value={"pg_backend": "gloo"})

    def collect():
        return (
            pg_probe()
            if tracer.allow_process_group_introspection
            else {"pg_backend": "cached-or-unknown"}
        )

    assert tracer.emit("first", collect=collect) is True
    assert tracer.emit("second", collect=collect) is False
    assert tracer.emit("third", collect=collect) is False

    assert pg_probe.call_count == 1
    assert log.info.call_count == 1
    log.warning.assert_called_once()
    assert [record["event"] for record in tracer.snapshot()["ring_tail"]] == [
        "first",
        "second",
        "third",
    ]


def test_concurrent_emit_keeps_unique_sequences_and_bounded_snapshot():
    tracer = PhaseTracer(
        enabled=True, max_events=0, every_n=1, ring_size=256, log=Mock()
    )
    barrier = threading.Barrier(8)

    def emit_worker(worker: int):
        barrier.wait()
        for offset in range(100):
            tracer.emit("concurrent", worker=worker, offset=offset)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(emit_worker, range(8)))

    snapshot = tracer.snapshot(tail=256)
    sequences = [record["seq"] for record in snapshot["ring_tail"]]
    assert len(sequences) == 256
    assert len(set(sequences)) == 256
    assert all(1 <= seq <= 800 for seq in sequences)
    assert tracer.last_marker() in snapshot["ring_tail"]


def test_watchdog_snapshot_format_contains_last_marker_and_bounded_tail():
    tracer = PhaseTracer(
        enabled=True, max_events=0, every_n=8, ring_size=256, log=Mock()
    )
    for seq in range(40):
        tracer.emit("watchdog_phase", marker=seq)

    payload = _payload(tracer.format_watchdog_snapshot(tail=32))
    assert payload["last_marker"]["marker"] == 39
    assert len(payload["ring_tail"]) == 32
    assert payload["ring_tail"][0]["marker"] == 8
    assert payload["ring_tail"][-1] == payload["last_marker"]


def test_watchdog_raw_write_is_bounded_and_ignores_oserror():
    tracer = PhaseTracer(
        enabled=True, max_events=0, every_n=8, ring_size=256, log=Mock()
    )
    tracer.emit("last", marker="x" * 1000)
    with (
        patch.object(phase_trace.select, "select", return_value=([], [2], [])),
        patch.object(phase_trace.os, "write") as write,
    ):
        tracer.write_watchdog_snapshot(tail=32)
    fd, payload = write.call_args.args
    assert fd == 2
    assert payload.startswith(b"PHASE_TRACE_WATCHDOG ")
    assert len(payload) <= phase_trace._WATCHDOG_MAX_BYTES

    with (
        patch.object(phase_trace.select, "select", return_value=([], [2], [])),
        patch.object(phase_trace.os, "write", side_effect=OSError),
    ):
        tracer.write_watchdog_snapshot(tail=32)

    with (
        patch.object(phase_trace.select, "select", return_value=([], [], [])),
        patch.object(phase_trace.os, "write") as write,
    ):
        tracer.write_watchdog_snapshot(tail=32)
    write.assert_not_called()


def test_environment_defaults_are_low_overhead_and_overrideable():
    assert environ.envs.SGLANG_DEBUG_PP_DP_PHASE_TRACE.default is False
    assert environ.envs.SGLANG_DEBUG_PP_DP_PHASE_TRACE_MAX_EVENTS.default == 4096
    assert environ.envs.SGLANG_DEBUG_PP_DP_PHASE_TRACE_EVERY_N.default == 8
    assert environ.envs.SGLANG_DEBUG_PP_DP_PHASE_TRACE_RING_SIZE.default == 256

    with environ.envs.SGLANG_DEBUG_PP_DP_PHASE_TRACE_EVERY_N.override(3):
        assert environ.envs.SGLANG_DEBUG_PP_DP_PHASE_TRACE_EVERY_N.get() == 3


def test_scheduler_watchdog_dump_includes_snapshot_without_sleep():
    invariant_checker_path = (
        _ROOT / "python/sglang/srt/managers/scheduler_components/invariant_checker.py"
    )
    source = invariant_checker_path.read_text()
    function_start = source.index("def format_scheduler_watchdog_dump(")
    function_end = source.index("\ndef create_scheduler_watchdog(", function_start)
    namespace = {
        "phase_tracer": SimpleNamespace(write_watchdog_snapshot=Mock()),
        "Scheduler": object,
    }
    exec(source[function_start:function_end], namespace)

    snapshot = "SGLANG_PP_DP_PHASE_TRACE_WATCHDOG snapshot"
    namespace["phase_tracer"].format_watchdog_snapshot = Mock(return_value=snapshot)
    scheduler = SimpleNamespace(
        is_initializing=False,
        invariant_checker=SimpleNamespace(
            _check_all_pools=Mock(return_value=(False, ["pools=ok"]))
        ),
        pool_stats_observer=SimpleNamespace(get_pool_stats=Mock(return_value=object())),
        cur_batch_for_debug=SimpleNamespace(
            batch_size=Mock(return_value=2), reqs=["r0", "r1"]
        ),
    )

    output = namespace["format_scheduler_watchdog_dump"](scheduler)

    assert "batch_size()=2" in output
    assert "pools=ok" in output
    assert output.endswith(snapshot)
    namespace["phase_tracer"].format_watchdog_snapshot.assert_called_once_with(tail=32)

    scheduler.is_initializing = True
    output = namespace["format_scheduler_watchdog_dump"](scheduler)
    assert "scheduler.is_initializing=True" in output
    assert output.endswith(snapshot)


def test_scheduler_watchdog_dump_keeps_snapshot_when_invariant_dump_fails():
    invariant_checker_path = (
        _ROOT / "python/sglang/srt/managers/scheduler_components/invariant_checker.py"
    )
    source = invariant_checker_path.read_text()
    function_start = source.index("def format_scheduler_watchdog_dump(")
    function_end = source.index("\ndef create_scheduler_watchdog(", function_start)
    namespace = {
        "Scheduler": object,
        "phase_tracer": SimpleNamespace(
            format_watchdog_snapshot=Mock(return_value="trace-snapshot"),
            write_watchdog_snapshot=Mock(),
        ),
    }
    exec(source[function_start:function_end], namespace)
    scheduler = SimpleNamespace(
        is_initializing=False,
        invariant_checker=SimpleNamespace(
            _check_all_pools=Mock(side_effect=RuntimeError("tensor unavailable"))
        ),
        pool_stats_observer=SimpleNamespace(get_pool_stats=Mock(return_value=object())),
        cur_batch_for_debug=None,
    )

    output = namespace["format_scheduler_watchdog_dump"](scheduler)

    assert "scheduler invariant dump failed: <RuntimeError>" in output
    assert output.endswith("trace-snapshot")


def test_phase_tracer_formats_mixed_idle_global_votes_from_cpu_metadata():
    log = Mock()
    tracer = PhaseTracer(enabled=True, max_events=1, every_n=1, log=log)
    assert tracer.emit(
        "dp_all_gather_exit",
        local_tokens=4,
        global_tokens=[4, 0],
        global_forward_modes=["DECODE", "IDLE"],
        global_decode_graph_vote=True,
        transport="cpu",
    )
    payload = _payload(log.info.call_args.args[0])
    assert payload["global_tokens"] == [4, 0]
    assert payload["global_forward_modes"] == ["DECODE", "IDLE"]
    assert payload["transport"] == "cpu"
