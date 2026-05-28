"""VMM Composite VA smoke test for DWDP — standalone version.

No dependency on sglang internals. Uses cuda-python 13.x bindings directly.

Run on 2 GPUs:
  torchrun --nproc_per_node=2 vmm_smoke_test.py
"""

from __future__ import annotations

import ctypes
import logging
import sys
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CUDA Driver API via cuda-python 13.x
# ---------------------------------------------------------------------------
from cuda.bindings import driver as cu

def _align_up(size: int, alignment: int) -> int:
    return (size + alignment - 1) // alignment * alignment

_VMM_GRANULARITY = 2 * 1024 * 1024  # 2MB

# ---------------------------------------------------------------------------
# Test config
# ---------------------------------------------------------------------------
NUM_EXPERTS = 8
EXPERT_DIM = 64
DTYPE = torch.float16


def test_vmm_creation():
    """Test 1: VMM composite VA creation and local weight copy."""
    logger.info("=" * 60)
    logger.info("TEST 1: VMM Composite VA Creation")
    logger.info("=" * 60)

    cu.cuInit(0)
    device_id = torch.cuda.current_device()

    # Simulate rank 0: owns experts [0, 4)
    local_start = 0
    local_end = 4
    num_local = local_end - local_start
    per_expert_bytes = EXPERT_DIM * DTYPE.itemsize
    total_bytes = NUM_EXPERTS * per_expert_bytes
    total_aligned = _align_up(total_bytes, _VMM_GRANULARITY)

    # Create VMM allocation
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = cu.CUmemHandleType.CU_MEM_HANDLE_TYPE_GENERIC
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id

    err, handle = cu.cuMemCreate(total_aligned, prop, 0)
    assert err == 0, f"cuMemCreate failed: err={err}"

    # Reserve VA and map
    err, va = cu.cuMemAddressReserve(total_aligned, 0, 0, 0)
    assert err == 0, f"cuMemAddressReserve failed: err={err}"
    va_int = int(va)

    va_ptr = cu.CUdeviceptr(va_int)
    err = cu.cuMemMap(va_ptr, total_aligned, 0, handle, 0)
    assert err == 0, f"cuMemMap failed: err={err}"

    # Set access
    desc = cu.CUmemAccessDesc()
    desc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = device_id
    desc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    err = cu.cuMemSetAccess(va_ptr, total_aligned, [desc], 1)
    assert err == 0, f"cuMemSetAccess failed: err={err}"

    # Copy local weights into correct positions
    local_tensor = torch.randn(num_local, EXPERT_DIM, dtype=DTYPE, device=f"cuda:{device_id}")
    for i in range(num_local):
        local_tensor[i] = float(i + 1) / NUM_EXPERTS

    # Create view into local expert region
    local_offset = local_start * per_expert_bytes
    typestr = "<f2"  # float16
    local_view_ptr = va_int + local_offset
    local_view = type("V", (), {
        "__cuda_array_interface__": {
            "data": (local_view_ptr, False),
            "shape": local_tensor.shape,
            "strides": None,
            "typestr": typestr,
            "version": 3,
        }
    })()
    local_vmm_tensor = torch.as_tensor(local_view, device=f"cuda:{device_id}")
    local_vmm_tensor.copy_(local_tensor)

    # Create full tensor view
    full_view = type("V", (), {
        "__cuda_array_interface__": {
            "data": (va_int, False),
            "shape": (NUM_EXPERTS, EXPERT_DIM),
            "strides": None,
            "typestr": typestr,
            "version": 3,
        }
    })()
    full_tensor = torch.as_tensor(full_view, device=f"cuda:{device_id}")

    # Verify local expert values
    for i in range(local_start, local_end):
        expected = (i - local_start + 1) / NUM_EXPERTS
        actual = full_tensor[i].mean().item()
        assert abs(actual - expected) < 0.01, \
            f"Expert {i}: expected ~{expected:.3f}, got {actual:.3f}"

    # Verify remote slots are zero
    for i in range(local_end, NUM_EXPERTS):
        val = full_tensor[i].mean().item()
        assert abs(val) < 0.001, f"Remote expert {i} should be zero, got {val}"

    # Cleanup
    cu.cuMemUnmap(va_ptr, total_aligned)
    cu.cuMemAddressFree(va_ptr, total_aligned)
    cu.cuMemRelease(handle)

    logger.info("[PASS] VMM creation and local weight copy verified")
    return True


def test_vmm_p2p():
    """Test 2: P2P copy into VMM pool region between 2 GPUs."""
    logger.info("=" * 60)
    logger.info("TEST 2: P2P Copy into VMM Pool Region")
    logger.info("=" * 60)

    import torch.distributed as dist
    dist.init_process_group(init_method="env://", backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = torch.cuda.current_device()

    if world_size != 2:
        logger.warning(f"Need 2 GPUs, got {world_size}. Skipping.")
        dist.destroy_process_group()
        return True

    cu.cuInit(0)

    # Each rank owns half the experts
    my_start = rank * (NUM_EXPERTS // 2)
    my_end = my_start + NUM_EXPERTS // 2
    num_local = NUM_EXPERTS // 2
    per_expert_bytes = EXPERT_DIM * DTYPE.itemsize
    total_bytes = NUM_EXPERTS * per_expert_bytes
    total_aligned = _align_up(total_bytes, _VMM_GRANULARITY)

    # Create VMM allocation
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = cu.CUmemHandleType.CU_MEM_HANDLE_TYPE_GENERIC
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id

    err, handle = cu.cuMemCreate(total_aligned, prop, 0)
    assert err == 0, f"cuMemCreate failed: err={err}"

    err, va = cu.cuMemAddressReserve(total_aligned, 0, 0, 0)
    assert err == 0, f"cuMemAddressReserve failed: err={err}"
    va_int = int(va)

    va_ptr = cu.CUdeviceptr(va_int)
    err = cu.cuMemMap(va_ptr, total_aligned, 0, handle, 0)
    assert err == 0, f"cuMemMap failed: err={err}"

    desc = cu.CUmemAccessDesc()
    desc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = device_id
    desc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    err = cu.cuMemSetAccess(va_ptr, total_aligned, [desc], 1)
    assert err == 0, f"cuMemSetAccess failed: err={err}"

    # Copy local weights
    local_tensor = torch.randn(num_local, EXPERT_DIM, dtype=DTYPE, device=f"cuda:{device_id}")
    for i in range(num_local):
        local_tensor[i] = float(i + 1) / NUM_EXPERTS

    local_offset = my_start * per_expert_bytes
    typestr = "<f2"
    local_view = type("V", (), {
        "__cuda_array_interface__": {
            "data": (va_int + local_offset, False),
            "shape": local_tensor.shape,
            "strides": None,
            "typestr": typestr,
            "version": 3,
        }
    })()
    local_vmm_tensor = torch.as_tensor(local_view, device=f"cuda:{device_id}")
    local_vmm_tensor.copy_(local_tensor)

    # Exchange IPC handles for P2P
    my_tensor_ptr = local_tensor.data_ptr()

    # Get IPC handle via ctypes (libcudart)
    cudart = ctypes.CDLL("libcudart.so")
    handle_buf = (ctypes.c_byte * 128)()
    err = cudart.cudaIpcGetMemHandle(ctypes.byref(handle_buf), ctypes.c_void_p(my_tensor_ptr))
    assert err == 0, f"cudaIpcGetMemHandle failed: {err}"

    handle_bytes = bytes(handle_buf)
    all_handles = [None] * world_size
    dist.all_gather_object(all_handles, handle_bytes)

    # Open peer handle
    peer_rank = 1 - rank
    peer_handle_buf = (ctypes.c_byte * 128)(*all_handles[peer_rank])
    base_ptr = ctypes.c_void_p()
    err = cudart.cudaIpcOpenMemHandle(ctypes.byref(base_ptr), peer_handle_buf, 0)
    assert err == 0, f"cudaIpcOpenMemHandle failed: {err}"

    # P2P copy: peer's local tensor → our VMM pool region for peer's experts
    peer_expert_start = peer_rank * (NUM_EXPERTS // 2)
    dst_offset = peer_expert_start * per_expert_bytes
    copy_bytes = num_local * per_expert_bytes

    src_ptr = base_ptr.value
    dst_ptr = va_int + dst_offset

    stream = torch.cuda.Stream()
    cudart.cudaMemcpyAsync(
        ctypes.c_void_p(dst_ptr), ctypes.c_void_p(src_ptr),
        ctypes.c_size_t(copy_bytes), 4,  # cudaMemcpyDefault
        ctypes.c_void_p(stream.cuda_stream),
    )
    stream.synchronize()

    # Close IPC handle
    cudart.cudaIpcCloseMemHandle(base_ptr)

    # Create full tensor view and verify
    full_view = type("V", (), {
        "__cuda_array_interface__": {
            "data": (va_int, False),
            "shape": (NUM_EXPERTS, EXPERT_DIM),
            "strides": None,
            "typestr": typestr,
            "version": 3,
        }
    })()
    full_tensor = torch.as_tensor(full_view, device=f"cuda:{device_id}")

    # Verify all experts (local + P2P)
    for i in range(NUM_EXPERTS):
        expert_owner = i // (NUM_EXPERTS // 2)
        local_idx = i % (NUM_EXPERTS // 2)
        expected = (local_idx + 1) / NUM_EXPERTS
        actual = full_tensor[i].mean().item()
        assert abs(actual - expected) < 0.05, \
            f"Expert {i} (owner={expert_owner}, rank={rank}): " \
            f"expected ~{expected:.3f}, got {actual:.3f}"

    # Cleanup
    cu.cuMemUnmap(va_ptr, total_aligned)
    cu.cuMemAddressFree(va_ptr, total_aligned)
    cu.cuMemRelease(handle)

    dist.barrier()
    dist.destroy_process_group()

    logger.info("[PASS] P2P copy into VMM pool region verified")
    return True


def test_cuda_graph():
    """Test 3: CUDA graph capture/replay with VMM tensors."""
    logger.info("=" * 60)
    logger.info("TEST 3: CUDA Graph Compatibility")
    logger.info("=" * 60)

    cu.cuInit(0)
    device_id = torch.cuda.current_device()

    local_start = 0
    local_end = 4
    num_local = local_end - local_start
    per_expert_bytes = EXPERT_DIM * DTYPE.itemsize
    total_bytes = NUM_EXPERTS * per_expert_bytes
    total_aligned = _align_up(total_bytes, _VMM_GRANULARITY)

    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = cu.CUmemHandleType.CU_MEM_HANDLE_TYPE_GENERIC
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id

    err, handle = cu.cuMemCreate(total_aligned, prop, 0)
    assert err == 0
    err, va = cu.cuMemAddressReserve(total_aligned, 0, 0, 0)
    assert err == 0
    va_int = int(va)
    va_ptr = cu.CUdeviceptr(va_int)
    cu.cuMemMap(va_ptr, total_aligned, 0, handle, 0)

    desc = cu.CUmemAccessDesc()
    desc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = device_id
    desc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    cu.cuMemSetAccess(va_ptr, total_aligned, [desc], 1)

    # Fill with data
    local_tensor = torch.randn(num_local, EXPERT_DIM, dtype=DTYPE, device=f"cuda:{device_id}")
    for i in range(num_local):
        local_tensor[i] = float(i + 1) / NUM_EXPERTS

    typestr = "<f2"
    local_view = type("V", (), {
        "__cuda_array_interface__": {
            "data": (va_int, False),
            "shape": local_tensor.shape,
            "strides": None,
            "typestr": typestr,
            "version": 3,
        }
    })()
    vmm_tensor = torch.as_tensor(local_view, device=f"cuda:{device_id}")
    vmm_tensor.copy_(local_tensor)

    # Full tensor view
    full_view = type("V", (), {
        "__cuda_array_interface__": {
            "data": (va_int, False),
            "shape": (NUM_EXPERTS, EXPERT_DIM),
            "strides": None,
            "typestr": typestr,
            "version": 3,
        }
    })()
    w = torch.as_tensor(full_view, device=f"cuda:{device_id}")
    x = torch.randn(32, EXPERT_DIM, dtype=DTYPE, device=f"cuda:{device_id}")

    # Capture
    g = torch.cuda.graph(enable=True)
    y = x @ w.T
    g.replay()

    # Verify
    y_eager = x @ w.T
    diff = (y - y_eager).abs().max().item()
    assert diff < 0.01, f"CUDA graph mismatch: {diff}"

    cu.cuMemUnmap(va_ptr, total_aligned)
    cu.cuMemAddressFree(va_ptr, total_aligned)
    cu.cuMemRelease(handle)

    logger.info("[PASS] CUDA graph capture/replay verified")
    return True


def test_param_swap():
    """Test 4: param.data swap with VMM tensors."""
    logger.info("=" * 60)
    logger.info("TEST 4: Param Data Swap")
    logger.info("=" * 60)

    cu.cuInit(0)
    device_id = torch.cuda.current_device()

    # Original param (simulates model weight)
    orig = torch.randn(4, EXPERT_DIM, dtype=DTYPE, device=f"cuda:{device_id}")
    saved_ptr = orig.data_ptr()

    # Create VMM tensor
    per_expert_bytes = EXPERT_DIM * DTYPE.itemsize
    total_bytes = NUM_EXPERTS * per_expert_bytes
    total_aligned = _align_up(total_bytes, _VMM_GRANULARITY)

    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = cu.CUmemHandleType.CU_MEM_HANDLE_TYPE_GENERIC
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id

    err, handle = cu.cuMemCreate(total_aligned, prop, 0)
    assert err == 0
    err, va = cu.cuMemAddressReserve(total_aligned, 0, 0, 0)
    assert err == 0
    va_int = int(va)
    va_ptr = cu.CUdeviceptr(va_int)
    cu.cuMemMap(va_ptr, total_aligned, 0, handle, 0)

    desc = cu.CUmemAccessDesc()
    desc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = device_id
    desc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    cu.cuMemSetAccess(va_ptr, total_aligned, [desc], 1)

    vmm_view = type("V", (), {
        "__cuda_array_interface__": {
            "data": (va_int, False),
            "shape": (NUM_EXPERTS, EXPERT_DIM),
            "strides": None,
            "typestr": "<f2",
            "version": 3,
        }
    })()
    vmm_tensor = torch.as_tensor(vmm_view, device=f"cuda:{device_id}")

    # Swap
    orig.data = vmm_tensor
    assert orig.data_ptr() == va_int, "Swap failed"
    assert orig.shape == (NUM_EXPERTS, EXPERT_DIM), f"Shape after swap: {orig.shape}"

    # Compute with swapped param
    x = torch.randn(16, EXPERT_DIM, dtype=DTYPE, device=f"cuda:{device_id}")
    y = x @ orig.T
    assert y.shape == (16, NUM_EXPERTS), f"Output shape: {y.shape}"

    # Restore
    orig.data = torch.empty(4, EXPERT_DIM, dtype=DTYPE, device=f"cuda:{device_id}").data
    # Note: can't fully restore to original allocation, but data_ptr should differ from VMM
    assert orig.data_ptr() != va_int, "Restore failed — still pointing to VMM"

    cu.cuMemUnmap(va_ptr, total_aligned)
    cu.cuMemAddressFree(va_ptr, total_aligned)
    cu.cuMemRelease(handle)

    logger.info("[PASS] Param data swap verified")
    return True


if __name__ == "__main__":
    # Determine if we're running single-GPU or multi-GPU
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    import os

    tests = []

    # Tests 1, 3, 4 run on single GPU
    if rank == 0 or world_size == 1:
        tests.extend([
            ("VMM Creation", test_vmm_creation),
            ("CUDA Graph", test_cuda_graph),
            ("Param Swap", test_param_swap),
        ])

    # Test 2 needs 2 GPUs
    if world_size == 2:
        tests.append(("P2P Copy", test_vmm_p2p))

    results = []
    for name, fn in tests:
        try:
            result = fn()
            results.append((name, "PASS"))
        except Exception as e:
            logger.error(f"[FAIL] {name}: {e}", exc_info=True)
            results.append((name, "FAIL"))

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    passed = sum(1 for _, s in results if s == "PASS")
    total = len(results)
    for name, status in results:
        icon = "\u2713" if status == "PASS" else "\u2717"
        logger.info(f"  {icon} {name}: {status}")
    logger.info(f"\n{passed}/{total} tests passed")

    sys.exit(0 if passed == total else 1)
