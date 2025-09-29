import os
import unittest
from types import SimpleNamespace
import time


from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestPureDP(CustomTestCase):
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
                "16",
                "--enable-dp-attention",
                "--dp",
                "16",
                "--dist-backend",
                "mooncake",
                "--moe-a2a-backend",
                "mooncake",
                "--deepep-mode",
                "low_latency",
                "--chunked-prefill-size",
                "512",
                "--cuda-graph-max-bs",
                "128",
                "--max-running-requests",
                "512",
                "--mem-fraction-static",
                "0.95",
                "--disable-cuda-graph",
                "--disable-custom-all-reduce",
                "--enable-eplb",
                "--ep-num-redundant-experts",
                "96",
                "--enable-dp-lm-head",
                "--moe-dense-tp-size",
                "1",
                "--dist-init-addr",
                "10.5.55.7:5000",
                "--nnodes",
                "2",
                "--node-rank",
                "0",
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

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)
        print(f"{speed=:.2f}")

    def test_bs_1_elastic(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        for i in range(10):
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
                "4",
                "--enable-dp-attention",
                "--dp",
                "2",
                "--dist-backend",
                "mooncake",
                "--moe-a2a-backend",
                "mooncake",
                "--deepep-mode",
                "low_latency",
                "--chunked-prefill-size",
                "512",
                "--cuda-graph-max-bs",
                "128",
                "--max-running-requests",
                "256",
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
                "--dist-backend",
                "mooncake",
                "--moe-a2a-backend",
                "mooncake",
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
                "--dist-backend",
                "mooncake",
                "--moe-a2a-backend",
                "mooncake",
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
                "--dist-backend",
                "mooncake",
                "--moe-a2a-backend",
                "mooncake",
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


if __name__ == "__main__":
    unittest.main()
