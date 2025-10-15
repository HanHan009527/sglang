import os
import time
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_disaggregation_utils import get_rdma_devices_args
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

ib_devices = get_rdma_devices_args()


class _BaseTestMooncakeEp(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--enable-dp-attention",
                "--dp",
                "4",
                "--elastic-ep-backend",
                "mooncake",
                "--mooncake-ib-device",
                ib_devices,
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "low_latency",
                "--chunked-prefill-size",
                "512",
                "--cuda-graph-max-bs",
                "128",
                "--max-running-requests",
                "512",
                "--mem-fraction-static",
                "0.5",
                "--disable-custom-all-reduce",
                "--enable-eplb",
                "--ep-num-redundant-experts",
                "24",
                "--enable-dp-lm-head",
                "--moe-dense-tp-size",
                "1",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        os.system("pkill -f sglang::scheduler_DP0_TP0_EP0")
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(metrics)

        self.assertGreater(metrics["accuracy"], 0.60)

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)
        print(f"{speed=:.2f}")

    def test_bs_1_fault_tolerance(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)
        print(f"{speed=:.2f}")
        os.system("pkill -f sglang::scheduler_DP0_TP0_EP0")
        acc_length, speed = send_one_prompt(args)
        print(f"{speed=:.2f}")
        acc_length, speed = send_one_prompt(args)
        print(f"{speed=:.2f}")
        acc_length, speed = send_one_prompt(args)
        print(f"{speed=:.2f}")
        acc_length, speed = send_one_prompt(args)
        print(f"{speed=:.2f}")
        acc_length, speed = send_one_prompt(args)
        print(f"{speed=:.2f}")
        acc_length, speed = send_one_prompt(args)
        print(f"{speed=:.2f}")


class TestHybridDPTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "8",
                "--enable-dp-attention",
                "--dp",
                "2",
                "--elastic-ep-backend",
                "mooncake",
                "--mooncake-ib-device",
ib_devices,                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "low_latency",
                "--chunked-prefill-size",
                "512",
                "--cuda-graph-max-bs",
                "128",
                "--mem-fraction-static",
                "0.7",
                "--max-running-requests",
                "256",
                "--disable-custom-all-reduce",
                "--enable-eplb",
                "--ep-num-redundant-experts",
                "24",
                "--enable-dp-lm-head",
                "--moe-dense-tp-size",
                "1",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        # os.system("pkill -f sglang::scheduler_DP0_TP0_EP0")
        os.system("pkill -f sglang::scheduler_DP1_TP2_EP2")
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(metrics)

        self.assertGreater(metrics["accuracy"], 0.60)


class TestTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--elastic-ep-backend",
                "mooncake",
                "--mooncake-ib-device",
ib_devices,                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "low_latency",
                "--chunked-prefill-size",
                "512",
                "--cuda-graph-max-bs",
                "128",
                "--max-running-requests",
                "128",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(metrics)

        self.assertGreater(metrics["accuracy"], 0.60)


class TestNoGatherdBuffer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--enable-dp-attention",
                "--dp",
                "4",
                "--moe-dense-tp-size",
                "1",
                "--enable-dp-lm-head",
                "--elastic-ep-backend",
                "mooncake",
                "--mooncake-ib-device",
ib_devices,                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "low_latency",
                "--chunked-prefill-size",
                "512",
                "--cuda-graph-max-bs",
                "32",
                "--max-running-requests",
                "512",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(metrics)

        self.assertGreater(metrics["accuracy"], 0.60)


class TestTBO(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--enable-dp-attention",
                "--dp",
                "4",
                "--moe-dense-tp-size",
                "1",
                "--elastic-ep-backend",
                "mooncake",
                "--mooncake-ib-device",
ib_devices,                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "low_latency",
                "--chunked-prefill-size",
                "512",
                "--enable-two-batch-overlap",
                "--cuda-graph-max-bs",
                "128",
                "--max-running-requests",
                "512",
            ],
        )


class TestTBO(_BaseTestMooncakeEp):
    extra_args = [
        "--tp",
        "4",
        "--enable-dp-attention",
        "--dp",
        "4",
        "--moe-dense-tp-size",
        "1",
        "--enable-two-batch-overlap",
    ]


class TestMooncakeWitchEPLB(_BaseTestMooncakeEp):
    extra_args = [
        "--tp",
        "4",
        "--enable-dp-attention",
        "--dp",
        "4",
        "--moe-dense-tp-size",
        "1",
        "--enable-two-batch-overlap",
        "--enable-eplb",
        "--ep-num-redundant-experts",
        "4",
        "--eplb-rebalance-num-iterations",
        "50",
        "--expert-distribution-recorder-buffer-size",
        "50",
        "--expert-distribution-recorder-mode",
        "stat",
        "--ep-dispatch-algorithm",
        "static",
    ]


if __name__ == "__main__":
    unittest.main()
