"""CPU contracts for DeepGEMM MegaMoE SM-count selection."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers.moe import mega_moe  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402


class FakeDeepGemm:
    def __init__(self, num_sms: int):
        self._num_sms = num_sms
        self.set_num_sms_calls = []

    def get_num_sms(self) -> int:
        return self._num_sms

    def set_num_sms(self, num_sms: int) -> None:
        self.set_num_sms_calls.append(num_sms)
        self._num_sms = num_sms


class TestMegaMoEDeepGemmNumSms(CustomTestCase):
    def test_h20_reserves_two_sms_and_restores(self):
        deep_gemm = FakeDeepGemm(num_sms=78)

        with (
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_SMS.override(0),
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.override(2),
            mega_moe._configure_mega_moe_deep_gemm_num_sms(deep_gemm) as num_sms,
        ):
            self.assertEqual(num_sms, 76)
            self.assertEqual(deep_gemm.get_num_sms(), 76)

        self.assertEqual(deep_gemm.get_num_sms(), 78)
        self.assertEqual(deep_gemm.set_num_sms_calls, [76, 78])

    def test_restores_original_count_when_forward_raises(self):
        deep_gemm = FakeDeepGemm(num_sms=78)

        with self.assertRaisesRegex(RuntimeError, "boom"):
            with (
                envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_SMS.override(0),
                envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.override(2),
                mega_moe._configure_mega_moe_deep_gemm_num_sms(deep_gemm),
            ):
                raise RuntimeError("boom")

        self.assertEqual(deep_gemm.get_num_sms(), 78)
        self.assertEqual(deep_gemm.set_num_sms_calls, [76, 78])

    def test_explicit_count_wins_and_is_rounded_down_to_even(self):
        deep_gemm = FakeDeepGemm(num_sms=78)

        with (
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_SMS.override(75),
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.override(20),
            mega_moe._configure_mega_moe_deep_gemm_num_sms(deep_gemm) as num_sms,
        ):
            self.assertEqual(num_sms, 74)

        self.assertEqual(deep_gemm.set_num_sms_calls, [74, 78])

    def test_zero_reserve_keeps_current_count(self):
        deep_gemm = FakeDeepGemm(num_sms=78)

        with (
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_SMS.override(0),
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.override(0),
            mega_moe._configure_mega_moe_deep_gemm_num_sms(deep_gemm) as num_sms,
        ):
            self.assertEqual(num_sms, 78)

        self.assertEqual(deep_gemm.set_num_sms_calls, [])

    def test_nested_budget_reserves_from_current_count(self):
        deep_gemm = FakeDeepGemm(num_sms=78)
        deep_gemm.set_num_sms(70)

        with (
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_SMS.override(0),
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.override(2),
            mega_moe._configure_mega_moe_deep_gemm_num_sms(deep_gemm) as num_sms,
        ):
            self.assertEqual(num_sms, 68)
            self.assertEqual(deep_gemm.get_num_sms(), 68)

        self.assertEqual(deep_gemm.get_num_sms(), 70)
        self.assertEqual(deep_gemm.set_num_sms_calls, [70, 68, 70])

    def test_forward_runs_routed_path_under_reserved_budget(self):
        deep_gemm = FakeDeepGemm(num_sms=78)
        moe = SimpleNamespace(
            alt_stream=None,
            num_fused_shared_experts=0,
            _forward_shared_experts=mock.Mock(return_value=None),
        )
        # The zero-token collective path still invokes both SM90 kernels, but
        # avoids unrelated router/EPLB runtime-context setup in this CPU test.
        hidden_states = SimpleNamespace(shape=(0, 16))
        routed_output = object()

        def check_budget(*args):
            self.assertEqual(deep_gemm.get_num_sms(), 76)
            return routed_output

        with (
            mock.patch.dict("sys.modules", {"deep_gemm": deep_gemm}),
            mock.patch.object(mega_moe, "get_is_capture_mode", return_value=False),
            mock.patch.object(
                mega_moe, "_run_mega_routed", side_effect=check_budget
            ) as run_routed,
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_SMS.override(0),
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.override(2),
        ):
            result = mega_moe.forward_mega_moe(moe, hidden_states)

        self.assertIs(result, routed_output)
        run_routed.assert_called_once()
        self.assertEqual(deep_gemm.get_num_sms(), 78)

    def test_sm90_launch_observes_reserved_budget(self):
        deep_gemm = FakeDeepGemm(num_sms=78)
        deep_gemm.get_symm_buffer_for_mega_moe = mock.Mock(return_value=object())
        moe = SimpleNamespace(
            alt_stream=None,
            num_fused_shared_experts=0,
            _forward_shared_experts=mock.Mock(return_value=None),
            gate=mock.Mock(return_value=torch.zeros((1, 2))),
            topk=mock.Mock(
                return_value=SimpleNamespace(
                    topk_ids=torch.zeros((1, 1), dtype=torch.int64),
                    topk_weights=torch.ones((1, 1)),
                )
            ),
            is_hash=False,
            layer_id=0,
            config=SimpleNamespace(
                hidden_size=16,
                num_experts_per_tok=1,
                moe_intermediate_size=8,
            ),
            experts=SimpleNamespace(num_experts=2),
        )
        ep_group = SimpleNamespace(device_group=SimpleNamespace())
        routed_output = object()

        def check_sm90_launch(*args):
            self.assertEqual(deep_gemm.get_num_sms(), 76)
            return routed_output

        with (
            mock.patch.dict("sys.modules", {"deep_gemm": deep_gemm}),
            mock.patch.object(mega_moe, "_device_sm", 90),
            mock.patch.object(mega_moe, "get_is_capture_mode", return_value=False),
            mock.patch(
                "sglang.srt.distributed.parallel_state.get_moe_ep_group",
                return_value=ep_group,
            ),
            mock.patch.object(
                mega_moe, "run_sm90_mega_routed", side_effect=check_sm90_launch
            ) as run_sm90,
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_SMS.override(0),
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.override(2),
        ):
            result = mega_moe.forward_mega_moe(moe, torch.zeros((0, 16)))

        self.assertIs(result, routed_output)
        run_sm90.assert_called_once()
        self.assertEqual(deep_gemm.get_num_sms(), 78)


if __name__ == "__main__":
    unittest.main()
