"""Opt-in tracing for PP relay and DP metadata-sync phase ordering.

Diagnostic only. The tracer deliberately accepts only host-side metadata. It
never inspects a tensor, synchronizes a device, or performs distributed
communication, so it cannot change collective count, shape, or ordering.

Default off: every entry point returns immediately unless
``SGLANG_DEBUG_PP_DP_PHASE_TRACE`` is set, so the disabled path adds no
formatter, process-group, or logging work to the communication hot path.

Single writer: the module-level ``phase_tracer`` is emitted only from the
scheduler thread that runs PP relay and DP sync. Snapshots taken from a
watchdog thread read a bounded copy under the GIL and never mutate tracer
state, so no cross-thread lock is required on the hot path.
"""

from __future__ import annotations

import json
import logging
import os
import select
from collections import deque
from itertools import count
from typing import Any, Callable, Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_LOG_PREFIX = "SGLANG_PP_DP_PHASE_TRACE "
_WATCHDOG_PREFIX = "PHASE_TRACE_WATCHDOG "
_WATCHDOG_TAIL_EVENTS = 32
_WATCHDOG_MAX_BYTES = 4096
_PROCESS_GROUP_METADATA_CACHE: dict[int, tuple[Any, dict[str, Any]]] = {}


def _safe_value(
    value: Any, *, _seen: Optional[set[int]] = None, _depth: int = 0
) -> Any:
    """Convert metadata to JSON primitives without invoking tensor access."""
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and len(value) > 512:
            return value[:512] + "<truncated>"
        return value
    if _depth >= 4:
        return f"<{type(value).__name__}:max-depth>"
    if _seen is None:
        _seen = set()
    value_id = id(value)
    if value_id in _seen:
        return f"<{type(value).__name__}:cycle>"
    if isinstance(value, (list, tuple)):
        _seen.add(value_id)
        try:
            items = [
                _safe_value(item, _seen=_seen, _depth=_depth + 1) for item in value[:64]
            ]
            if len(value) > 64:
                items.append(f"<{len(value) - 64} more>")
            return items
        finally:
            _seen.discard(value_id)
    if isinstance(value, dict):
        _seen.add(value_id)
        try:
            result = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= 64:
                    result["<truncated>"] = f"<{len(value) - 64} more>"
                    break
                result[str(key)[:256]] = _safe_value(
                    item, _seen=_seen, _depth=_depth + 1
                )
            return result
        finally:
            _seen.discard(value_id)
    return f"<{type(value).__name__}>"


def _make_phase_trace_record(
    event: str, seq: int, fields: dict[str, Any]
) -> dict[str, Any]:
    record = {"event": event, "seq": seq}
    record.update({key: _safe_value(value) for key, value in fields.items()})
    return record


def _format_record(prefix: str, record: dict[str, Any]) -> str:
    return prefix + json.dumps(
        record, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def format_phase_trace(event: str, seq: int, fields: dict[str, Any]) -> str:
    """Format one deterministic, single-line JSON trace record."""
    return _format_record(_LOG_PREFIX, _make_phase_trace_record(event, seq, fields))


def format_phase_trace_snapshot(snapshot: dict[str, Any]) -> str:
    """Format a bounded snapshot for scheduler watchdog diagnostics."""
    return _format_record(_WATCHDOG_PREFIX, _safe_value(snapshot))


def clear_process_group_metadata_cache() -> None:
    """Clear the process-local metadata cache (primarily for tests)."""
    _PROCESS_GROUP_METADATA_CACHE.clear()


def describe_process_group(
    group,
    *,
    dist=None,
    known_members: Optional[list[int]] = None,
    known_rank: Optional[int] = None,
    known_backend: Optional[str] = None,
    allow_introspection: bool = True,
) -> dict[str, Any]:
    """Best-effort cached process-group metadata with no communication."""
    cache_key = id(group) if group is not None else None
    if cache_key is not None:
        cached = _PROCESS_GROUP_METADATA_CACHE.get(cache_key)
        if cached is not None and cached[0] is group:
            return dict(cached[1])

    result = {
        "pg_backend": known_backend or "unknown",
        "pg_members": list(known_members) if known_members is not None else None,
        "pg_rank": known_rank,
        "pg_size": len(known_members) if known_members is not None else None,
    }
    if group is None:
        return result
    if dist is None or not allow_introspection:
        return result

    try:
        result["pg_backend"] = str(dist.get_backend(group))
    except Exception:
        pass
    try:
        result["pg_size"] = dist.get_world_size(group)
    except Exception:
        pass
    try:
        result["pg_rank"] = dist.get_rank(group)
    except Exception:
        pass
    if result["pg_members"] is None:
        try:
            result["pg_members"] = list(dist.get_process_group_ranks(group))
        except Exception:
            pass
    # Process groups are long-lived. Keeping a strong reference prevents an id
    # from being reused for a different group and makes subsequent emits a
    # plain dictionary lookup rather than repeated torch.distributed calls.
    _PROCESS_GROUP_METADATA_CACHE[cache_key] = (group, dict(result))
    return result


def batch_phase_fields(batch) -> dict[str, Any]:
    """Read only Python-side batch metadata suitable for phase tracing."""
    if batch is None:
        return {
            "active": False,
            "batch_id": None,
            "batch_size": 0,
            "forward_mode": None,
            "batch_global_decode_graph_vote": None,
            "batch_global_draft_graph_vote": None,
            "batch_global_tokens": None,
            "idle": True,
        }

    forward_mode = getattr(batch, "forward_mode", None)
    mode_name = getattr(forward_mode, "name", None)
    if mode_name is None and isinstance(forward_mode, (str, int)):
        mode_name = forward_mode

    idle = False
    is_idle = getattr(forward_mode, "is_idle", None)
    if callable(is_idle):
        idle = bool(is_idle())

    reqs = getattr(batch, "reqs", None)
    batch_size = len(reqs) if isinstance(reqs, (list, tuple)) else None
    return {
        "active": not idle,
        "batch_id": hex(id(batch)),
        "batch_size": batch_size,
        "forward_mode": mode_name,
        "batch_global_decode_graph_vote": getattr(batch, "can_run_dp_cuda_graph", None),
        "batch_global_draft_graph_vote": getattr(
            batch, "can_run_dp_draft_cuda_graph", None
        ),
        "batch_global_tokens": getattr(batch, "global_num_tokens", None),
        "idle": idle,
    }


class PhaseTracer:
    """A process-local bounded ring and sampled phase-order logger."""

    def __init__(
        self,
        *,
        enabled: bool,
        max_events: int,
        every_n: int,
        ring_size: int = 256,
        log: logging.Logger = logger,
    ) -> None:
        self.enabled = enabled
        self.max_events = max(0, max_events)
        self.every_n = max(1, every_n)
        self.log = log
        self._sequence = count(1)
        self._log_slots = count(1)
        self._ring = deque(maxlen=max(1, ring_size))
        self._logging_exhausted = self.max_events == 0

    @property
    def allow_process_group_introspection(self) -> bool:
        return self.enabled and not self._logging_exhausted

    def emit(
        self,
        event: str,
        *,
        collect: Optional[Callable[[], dict[str, Any]]] = None,
        **fields: Any,
    ) -> bool:
        if not self.enabled:
            return False

        try:
            if collect is not None:
                fields = {**collect(), **fields}
            # next() and deque.append() are atomic while the CPython GIL is
            # held. Keep the communication hot path lock-free.
            seq = next(self._sequence)
            record = _make_phase_trace_record(event, seq, fields)
            self._ring.append(record)

            if seq % self.every_n != 0:
                return False

            log_slot = next(self._log_slots)
            if log_slot > self.max_events:
                self._logging_exhausted = True
                if log_slot == self.max_events + 1:
                    # The unique counter slot guarantees warning-once even
                    # when several scheduler threads emit concurrently.
                    try:
                        self.log.warning(
                            "%slimit_reached max_events=%s every_n=%s",
                            _LOG_PREFIX,
                            self.max_events,
                            self.every_n,
                        )
                    except Exception:
                        pass
                return False

            if log_slot == self.max_events:
                self._logging_exhausted = True
            try:
                self.log.info(_format_record(_LOG_PREFIX, record))
            except Exception:
                return False
            return True
        except Exception:
            # Tracing is diagnostic only: field properties, recursive or
            # hostile containers, formatting, and storage must never escape
            # into scheduler execution.
            return False

    def last_marker(self) -> Optional[dict[str, Any]]:
        """Return a copy of the latest marker without touching torch/CUDA."""
        try:
            marker = self._ring[-1]
        except IndexError:
            marker = None
        return None if marker is None else dict(marker)

    def snapshot(self, tail: int = _WATCHDOG_TAIL_EVENTS) -> dict[str, Any]:
        """Return a bounded GIL-safe copy for a watchdog or crash dump."""
        tail = max(0, min(tail, self._ring.maxlen))
        # deque.copy() runs under the GIL; converting this private copy cannot
        # race with scheduler-thread appends to the live deque.
        records = list(self._ring.copy())
        marker = records[-1] if records else None
        if tail == 0:
            records = []
        else:
            records = records[-tail:]
        return {
            "enabled": self.enabled,
            "last_marker": None if marker is None else dict(marker),
            "ring_size": self._ring.maxlen,
            "ring_tail": [dict(record) for record in records],
        }

    def format_watchdog_snapshot(self, tail: int = _WATCHDOG_TAIL_EVENTS) -> str:
        try:
            return format_phase_trace_snapshot(self.snapshot(tail=tail))
        except Exception as exc:
            # A diagnostic formatter must not crash the watchdog thread.
            return format_phase_trace_snapshot(
                {
                    "enabled": self.enabled,
                    "snapshot_error": f"<{type(exc).__name__}>",
                }
            )

    def watchdog_snapshot_bytes(
        self,
        tail: int = _WATCHDOG_TAIL_EVENTS,
        max_bytes: int = _WATCHDOG_MAX_BYTES,
    ) -> bytes:
        """Return a bounded payload suitable for direct stderr output."""
        try:
            payload = (self.format_watchdog_snapshot(tail=tail) + "\n").encode(
                "utf-8", errors="replace"
            )
            max_bytes = max(1, max_bytes)
            if len(payload) > max_bytes:
                payload = payload[: max_bytes - 1] + b"\n"
            return payload
        except Exception:
            return b"PHASE_TRACE_WATCHDOG snapshot_error\n"

    def write_watchdog_snapshot(self, tail: int = _WATCHDOG_TAIL_EVENTS) -> None:
        """Best-effort raw stderr dump independent of logging handlers."""
        try:
            # Avoid waiting behind a wedged or full stderr consumer. A single
            # PIPE_BUF-sized write follows only when fd 2 is immediately ready.
            _, writable, _ = select.select([], [2], [], 0)
            if writable:
                os.write(2, self.watchdog_snapshot_bytes(tail=tail))
        except OSError:
            pass
        except Exception:
            pass


# Process-local singleton emitted only by the scheduler thread; default off.
phase_tracer = PhaseTracer(
    enabled=envs.SGLANG_DEBUG_PP_DP_PHASE_TRACE.get(),
    max_events=envs.SGLANG_DEBUG_PP_DP_PHASE_TRACE_MAX_EVENTS.get(),
    every_n=envs.SGLANG_DEBUG_PP_DP_PHASE_TRACE_EVERY_N.get(),
    ring_size=envs.SGLANG_DEBUG_PP_DP_PHASE_TRACE_RING_SIZE.get(),
)
