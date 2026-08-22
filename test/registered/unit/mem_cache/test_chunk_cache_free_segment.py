"""Contract tests for ChunkCache request completion cleanup."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestChunkCacheFreeSegment(unittest.TestCase):
    def test_cache_finished_req_forwards_unprotected_segment(self):
        allocator = Mock()
        req_to_token = torch.arange(192, dtype=torch.int64).reshape(1, -1)
        cache = ChunkCache.__new__(ChunkCache)
        cache.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
        cache.token_to_kv_pool_allocator = allocator
        req = SimpleNamespace(req_pool_idx=0, cache_protected_len=64)

        cache.cache_finished_req(req, kv_len_to_handle=130)

        allocator.free_segment.assert_called_once()
        allocator.free.assert_not_called()
        freed_indices, = allocator.free_segment.call_args.args
        self.assertTrue(torch.equal(freed_indices, req_to_token[0, 64:130]))
        self.assertEqual(
            allocator.free_segment.call_args.kwargs,
            {"start_pos": req.cache_protected_len},
        )
        self.assertTrue(torch.equal(freed_indices, torch.arange(64, 130)))

    def test_grouped_paged_free_owns_representatives_before_row_reuse(self):
        page_size = 64
        allocator = PagedTokenToKVPoolAllocator(
            size=8 * page_size,
            page_size=page_size,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=True,
        )
        allocated = allocator.alloc(3 * page_size)
        self.assertIsNotNone(allocated)
        req_to_token = allocated.reshape(1, -1)
        original_pages = torch.unique(req_to_token[0, page_size:130] // page_size)
        cache = ChunkCache.__new__(ChunkCache)
        cache.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
        cache.token_to_kv_pool_allocator = allocator
        req = SimpleNamespace(req_pool_idx=0, cache_protected_len=page_size)

        allocator.free_group_begin()
        cache.cache_finished_req(req, kv_len_to_handle=130)
        req_to_token[0].zero_()
        allocator.free_group_end()

        self.assertTrue(
            torch.equal(torch.sort(allocator.release_pages)[0], original_pages)
        )
        self.assertNotIn(0, allocator.release_pages.tolist())


if __name__ == "__main__":
    unittest.main()
