# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed
import torch.nn.functional as F

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.eplb import eplb_algorithms
from sglang.srt.model_loader import get_model_architecture
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass
class ExpertLocationMetadata:
    physical_to_logical_map: torch.Tensor  # (layers, num_physical_experts)
    physical_to_logical_map_cpu: torch.Tensor
    logical_to_all_physical_map: torch.Tensor  # (layers, num_logical_experts, X)
    logical_to_all_physical_map_num_valid: torch.Tensor  # (layers, num_logical_experts)
    # (layers, num_logical_experts)
    logical_to_rank_dispatch_physical_map: Optional[torch.Tensor]

    # -------------------------------- properties ------------------------------------

    @property
    def num_layers(self) -> int:
        return self.physical_to_logical_map.shape[0]

    @property
    def num_physical_experts(self) -> int:
        return self.physical_to_logical_map.shape[1]

    @property
    def num_local_physical_experts(self) -> int:
        ans, remainder = divmod(self.num_physical_experts, self.ep_size)
        assert remainder == 0
        return ans

    @property
    def num_logical_experts(self) -> int:
        return self.logical_to_all_physical_map.shape[1]

    @property
    def ep_size(self):
        # TODO change when EP size != world size
        return torch.distributed.get_world_size()

    def __post_init__(self):
        num_layers_0, num_physical_experts_0 = self.physical_to_logical_map.shape
        num_layers_1, num_logical_experts_0, num_physical_experts_1 = (
            self.logical_to_all_physical_map.shape
        )
        num_layers_2, num_logical_experts_1 = (
            self.logical_to_all_physical_map_num_valid.shape
        )
        assert num_layers_0 == num_layers_1 == num_layers_2
        assert num_logical_experts_0 == num_logical_experts_1
        assert num_physical_experts_0 == num_physical_experts_1

    # -------------------------------- construction ------------------------------------

    @staticmethod
    def init_trivial(server_args: ServerArgs, model_config: ModelConfig):
        """Trivial location - logical expert i corresponds to physical expert i"""
        common = ExpertLocationMetadata._init_common(server_args, model_config)
        num_physical_experts = common["num_physical_experts"]
        model_config_for_expert_location = common["model_config_for_expert_location"]
        num_layers = model_config_for_expert_location.num_layers
        num_logical_experts = model_config_for_expert_location.num_logical_experts

        physical_to_logical_map = (
            torch.arange(0, num_physical_experts).repeat(num_layers, 1)
            % num_logical_experts
        )

        return ExpertLocationMetadata.init_by_mapping(
            server_args,
            model_config,
            physical_to_logical_map=physical_to_logical_map,
        )

    @staticmethod
    def init_by_mapping(
        server_args: ServerArgs,
        model_config: ModelConfig,
        physical_to_logical_map,
    ):
        if not isinstance(physical_to_logical_map, torch.Tensor):
            physical_to_logical_map = torch.tensor(physical_to_logical_map)
        physical_to_logical_map = physical_to_logical_map.to(server_args.device)

        common = ExpertLocationMetadata._init_common(server_args, model_config)
        model_config_for_expert_location = common["model_config_for_expert_location"]
        logical_to_all_physical_map = _compute_logical_to_all_physical_map(
            physical_to_logical_map,
            num_logical_experts=model_config_for_expert_location.num_logical_experts,
        )

        return ExpertLocationMetadata._init_raw(
            server_args=server_args,
            ep_size=common["ep_size"],
            physical_to_logical_map=physical_to_logical_map,
            logical_to_all_physical_map=logical_to_all_physical_map,
        )

    @staticmethod
    def init_by_eplb(
        server_args: ServerArgs, model_config: ModelConfig, logical_count: torch.Tensor
    ):
        if not isinstance(logical_count, torch.Tensor):
            logical_count = torch.tensor(logical_count)
        if len(logical_count.shape) == 2:
            logical_count = logical_count.unsqueeze(0)
        logical_count = logical_count.to(server_args.device)

        common = ExpertLocationMetadata._init_common(server_args, model_config)
        model_config_for_expert_location = common["model_config_for_expert_location"]
        num_physical_experts = common["num_physical_experts"]
        num_groups = model_config_for_expert_location.num_groups
        num_nodes = server_args.nnodes

        physical_to_logical_map, logical_to_all_physical_map, expert_count = (
            eplb_algorithms.rebalance_experts(
                tokens_per_expert=logical_count,
                num_physical_experts=num_physical_experts,
                num_local_physical_experts=num_physical_experts // common["ep_size"],
                num_groups=num_groups,
                num_nodes=num_nodes,
                algorithm=eplb_algorithms.compute_algorithm(
                    raw_algorithm=server_args.eplb_algorithm,
                    num_groups=num_groups,
                    num_nodes=num_nodes,
                ),
            )
        )

        return ExpertLocationMetadata._init_raw(
            server_args=server_args,
            ep_size=common["ep_size"],
            physical_to_logical_map=physical_to_logical_map.to(server_args.device),
            logical_to_all_physical_map=logical_to_all_physical_map.to(
                server_args.device
            ),
        )

    @staticmethod
    def _init_common(server_args: ServerArgs, model_config: ModelConfig):
        model_config_for_expert_location = (
            ModelConfigForExpertLocation.from_model_config(model_config)
        )

        num_physical_experts = (
            model_config_for_expert_location.num_logical_experts
            + server_args.ep_num_redundant_experts
        )
        ep_size = server_args.ep_size
        assert num_physical_experts % ep_size == 0
        num_local_physical_experts = num_physical_experts // ep_size

        return dict(
            model_config_for_expert_location=model_config_for_expert_location,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            ep_size=ep_size,
        )

    @staticmethod
    def _init_raw(
        server_args: ServerArgs,
        ep_size: int,
        physical_to_logical_map: torch.Tensor,
        logical_to_all_physical_map: torch.Tensor,
    ):
        _, num_physical_experts = physical_to_logical_map.shape

        logical_to_all_physical_map_padded = F.pad(
            logical_to_all_physical_map,
            (0, num_physical_experts - logical_to_all_physical_map.shape[-1]),
            value=-1,
        )

        logical_to_all_physical_map_num_valid = torch.count_nonzero(
            logical_to_all_physical_map != -1, dim=-1
        )

        avoid_rank_str = os.environ.get("SGLANG_EP_AVOID_RANK")
        avoid_rank = int(avoid_rank_str) if avoid_rank_str else -1

        return ExpertLocationMetadata(
            physical_to_logical_map=physical_to_logical_map,
            physical_to_logical_map_cpu=physical_to_logical_map.cpu(),
            logical_to_all_physical_map=logical_to_all_physical_map_padded,
            logical_to_all_physical_map_num_valid=logical_to_all_physical_map_num_valid,
            logical_to_rank_dispatch_physical_map=(
                (
                    (
                        compute_logical_to_rank_dispatch_physical_map_avoid_rank(
                            logical_to_all_physical_map=logical_to_all_physical_map,
                            num_gpus=ep_size,
                            num_physical_experts=num_physical_experts,
                            # TODO improve when we have real EP rank
                            ep_rank=torch.distributed.get_rank() % ep_size,
                            avoid_rank=avoid_rank,
                        )
                    )
                    if avoid_rank != -1
                    else compute_logical_to_rank_dispatch_physical_map(
                        logical_to_all_physical_map=logical_to_all_physical_map,
                        num_gpus=ep_size,
                        num_physical_experts=num_physical_experts,
                        # TODO improve when we have real EP rank
                        ep_rank=torch.distributed.get_rank() % ep_size,
                    )
                )
                if server_args.ep_dispatch_algorithm == "static"
                else None
            ),
        )

    # -------------------------------- mutation ------------------------------------

    def update(
        self,
        other: "ExpertLocationMetadata",
        update_layer_ids: List[int],
    ):
        for field in [
            "ep_size",
        ]:
            assert getattr(self, field) == getattr(other, field)

        for field in [
            "physical_to_logical_map",
            "physical_to_logical_map_cpu",
            "logical_to_all_physical_map",
            "logical_to_all_physical_map_num_valid",
            "logical_to_rank_dispatch_physical_map",
        ]:
            other_field = getattr(other, field)
            self_field = getattr(self, field)
            assert (other_field is not None) == (self_field is not None)
            if self_field is not None:
                mask_update = torch.tensor(
                    [i in update_layer_ids for i in range(self.num_layers)]
                )
                mask_update = mask_update.view(*([-1] + [1] * (self_field.dim() - 1)))
                mask_update = mask_update.to(self_field.device, non_blocking=True)
                self_field[...] = torch.where(mask_update, other_field, self_field)

    # -------------------------------- usage ------------------------------------

    def logical_to_all_physical(
        self, layer_id: int, logical_expert_id: int
    ) -> List[int]:
        return [
            physical_expert_id
            for physical_expert_id in self.logical_to_all_physical_map[
                layer_id, logical_expert_id
            ].tolist()
            if physical_expert_id != -1
        ]


_global_expert_location_metadata: Optional[ExpertLocationMetadata] = None


def get_global_expert_location_metadata():
    return _global_expert_location_metadata


def set_global_expert_location_metadata(value):
    global _global_expert_location_metadata
    assert _global_expert_location_metadata is None
    _global_expert_location_metadata = value


def _compute_logical_to_all_physical_map(
    physical_to_logical_map: torch.Tensor, num_logical_experts: int
):
    # This is rarely called, so we use for loops for maximum clarity

    num_layers, num_physical_experts = physical_to_logical_map.shape

    logical_to_all_physical_map = [
        [[] for _ in range(num_logical_experts)] for _ in range(num_layers)
    ]
    for layer_id in range(num_layers):
        for physical_expert_id in range(num_physical_experts):
            logical_expert_id = physical_to_logical_map[
                layer_id, physical_expert_id
            ].item()
            logical_to_all_physical_map[layer_id][logical_expert_id].append(
                physical_expert_id
            )

    logical_to_all_physical_map = _pad_nested_array(
        logical_to_all_physical_map, pad_value=-1
    )

    return torch.tensor(
        logical_to_all_physical_map, device=physical_to_logical_map.device
    )


def _pad_nested_array(arr, pad_value):
    max_len = max(len(inner) for outer in arr for inner in outer)
    padded = [
        [inner + [pad_value] * (max_len - len(inner)) for inner in outer]
        for outer in arr
    ]
    return padded


def compute_logical_to_rank_dispatch_physical_map_remote_first(
    logical_to_all_physical_map: torch.Tensor,
    num_gpus: int,
    num_physical_experts: int,
    ep_rank: int,
    seed: int = 42,
):
    """Computes a static dispatch map from logical to physical experts, prioritizing remote experts.

    This function creates a dispatch map where each (GPU, logical expert) pair is assigned a
    specific physical expert. The key difference from the default implementation is its preference
    for assigning tasks to experts on different GPUs (remote experts) to potentially improve
    workload distribution across the system, falling back to local experts only when no remote
    options are available.

    1.  **Remote-First Assignment**: For each GPU, it identifies all available physical experts
        located on other GPUs. If such experts exist, it selects one with the lowest current
        load to handle the request.
    2.  **Load Balancing**: It maintains a load counter for each physical expert to ensure that
        requests are distributed as evenly as possible among the available candidates.
    3.  **Local Fallback**: If a logical expert has no physical replicas on other GPUs, the
        algorithm will assign a local expert (from the same GPU) instead.
    4.  **Deterministic Tie-Breaking**: The process is made deterministic by using a fixed seed.
        When multiple experts have the same load, shuffling the candidates before selection
        ensures fair tie-breaking.

    Args:
        logical_to_all_physical_map (torch.Tensor): A 3D tensor mapping each logical expert
            to its physical replicas. Shape: `(num_layers, num_logical_experts, num_replicas)`.
        num_gpus (int): The total number of GPUs in the expert parallel group.
        num_physical_experts (int): The total number of physical experts.
        ep_rank (int): The rank of the current process within the expert parallel group.
        seed (int): A seed for the random number generator to ensure deterministic behavior.

    Returns:
        torch.Tensor: A 2D tensor for the current `ep_rank` that maps each logical expert
                      to a physical expert. Shape: `(num_layers, num_logical_experts)`.
    """
    r = random.Random(seed)

    # 计算每个GPU上的物理专家数量
    num_local_physical_experts = num_physical_experts // num_gpus
    # 获取映射表的维度信息
    num_layers, num_logical_experts, _ = logical_to_all_physical_map.shape
    dtype = logical_to_all_physical_map.dtype

    # 创建一个用于存储最终分派映射的张量，并用-1填充
    logical_to_rank_dispatch_physical_map = torch.full(
        size=(num_gpus, num_layers, num_logical_experts),
        fill_value=-1,
        dtype=dtype,
    )

    # 遍历每一层
    for layer_id in range(num_layers):
        # 遍历该层中的每一个逻辑专家
        for logical_expert_id in range(num_logical_experts):
            # 获取当前逻辑专家的所有物理副本ID
            candidate_physical_expert_ids = _logical_to_all_physical_raw(
                logical_to_all_physical_map, layer_id, logical_expert_id
            )
            # 获取当前逻辑专家在所有GPU上的分派映射视图
            output_partial = logical_to_rank_dispatch_physical_map[
                :, layer_id, logical_expert_id
            ]

            # 为每个物理专家初始化负载计数器
            load = {p_id: 0 for p_id in candidate_physical_expert_ids}

            # 遍历所有GPU，为每个GPU分配一个专家
            for gpu_id in range(num_gpus):
                # --- 远程优先选择阶段 ---
                # 找出所有不位于当前GPU上的物理专家（即远程专家）
                remote_experts = [
                    p_id
                    for p_id in candidate_physical_expert_ids
                    if _compute_gpu_id_of_physical_expert(
                        p_id, num_local_physical_experts
                    )
                    != gpu_id
                ]

                # 如果存在远程专家，则从远程专家中选择；否则，从所有候选专家中选择（本地回退）
                if remote_experts:
                    experts_to_choose_from = remote_experts
                else:
                    experts_to_choose_from = candidate_physical_expert_ids

                # 为了在负载相同时打破僵局，随机打乱候选专家列表
                r.shuffle(experts_to_choose_from)

                # --- 负载均衡选择 ---
                # 从候选专家中选择一个当前负载最低的专家
                chosen_expert = min(experts_to_choose_from, key=lambda p_id: load[p_id])

                # 将选中的专家分配给当前GPU
                output_partial[gpu_id] = chosen_expert
                # 更新被选中专家的负载计数
                load[chosen_expert] += 1

    # 断言确保所有条目都已被成功分配
    assert torch.all(logical_to_rank_dispatch_physical_map != -1)

    # 获取原始张量的设备信息
    device = logical_to_all_physical_map.device
    # 返回属于当前ep_rank的分派映射表，并移动到正确的设备上
    return logical_to_rank_dispatch_physical_map[ep_rank, :, :].to(device)


def compute_logical_to_rank_dispatch_physical_map_avoid_rank(
    logical_to_all_physical_map: torch.Tensor,
    num_gpus: int,
    num_physical_experts: int,
    ep_rank: int,
    avoid_rank: int,
    seed: int = 42,
):
    """计算一个静态分派映射表，避免向特定rank调度任务。

    此函数旨在创建一个分派映射，其中每个 (GPU, 逻辑专家) 对被分配一个特定的物理专家，
    同时避免将任务分配给 `avoid_rank` 上的专家。如果过滤后没有可用专家，则会回退到使用所有可用专家。

    1.  **避免特定Rank**: 对于每个逻辑专家，它会首先过滤掉位于 `avoid_rank` 上的所有物理专家。
    2.  **负载均衡**: 在剩余的专家中，它使用负载计数器来确保请求在可用的候选专家中均匀分配。
    3.  **回退机制**: 如果过滤后没有可用的专家（例如，所有专家都在 `avoid_rank` 上），
        该算法将回退到在所有候选专家（包括在 `avoid_rank` 上的专家）中进行选择，以确保任务能够被分配。
    4.  **确定性**: 整个过程通过固定的随机种子来保证确定性。

    Args:
        logical_to_all_physical_map (torch.Tensor): 一个三维张量，映射每个逻辑专家到其所有物理副本。
            形状为 `(num_layers, num_logical_experts, num_replicas)`。
        num_gpus (int): 专家并行组中的 GPU 总数。
        num_physical_experts (int): 物理专家的总数。
        ep_rank (int): 当前进程在专家并行组中的排名。
        avoid_rank (int): 需要避免调度的 GPU rank。
        seed (int): 用于随机数生成器的种子，以确保确定性行为。

    Returns:
        torch.Tensor: 一个二维张量，为当前的 `ep_rank` 映射每个逻辑专家到一个物理专家。
                      形状为 `(num_layers, num_logical_experts)`。
    """
    r = random.Random(seed)

    # 计算每个GPU上的物理专家数量
    num_local_physical_experts = num_physical_experts // num_gpus
    # 获取映射表的维度信息
    num_layers, num_logical_experts, _ = logical_to_all_physical_map.shape
    dtype = logical_to_all_physical_map.dtype

    # 创建一个用于存储最终分派映射的张量，并用-1填充
    logical_to_rank_dispatch_physical_map = torch.full(
        size=(num_gpus, num_layers, num_logical_experts),
        fill_value=-1,
        dtype=dtype,
    )

    # 遍历每一层
    for layer_id in range(num_layers):
        # 遍历该层中的每一个逻辑专家
        for logical_expert_id in range(num_logical_experts):
            # 获取当前逻辑专家的所有物理副本ID
            candidate_physical_expert_ids = _logical_to_all_physical_raw(
                logical_to_all_physical_map, layer_id, logical_expert_id
            )

            # 筛选出不位于 avoid_rank 上的专家
            experts_to_choose_from = [
                p_id
                for p_id in candidate_physical_expert_ids
                if _compute_gpu_id_of_physical_expert(p_id, num_local_physical_experts)
                != avoid_rank
            ]

            # 如果筛选后没有专家可用，则回退到使用所有候选专家
            if not experts_to_choose_from:
                experts_to_choose_from = candidate_physical_expert_ids

            # 获取当前逻辑专家在所有GPU上的分派映射视图
            output_partial = logical_to_rank_dispatch_physical_map[
                :, layer_id, logical_expert_id
            ]

            # 为每个物理专家初始化负载计数器
            load = {p_id: 0 for p_id in experts_to_choose_from}

            # 遍历所有GPU，为每个GPU分配一个专家
            for gpu_id in range(num_gpus):
                # 为了在负载相同时打破僵局，随机打乱候选专家列表
                shuffled_experts = list(experts_to_choose_from)
                r.shuffle(shuffled_experts)

                # 从候选专家中选择一个当前负载最低的专家
                chosen_expert = min(shuffled_experts, key=lambda p_id: load[p_id])

                # 将选中的专家分配给当前GPU
                output_partial[gpu_id] = chosen_expert
                # 更新被选中专家的负载计数
                load[chosen_expert] += 1

    # 断言确保所有条目都已被成功分配
    assert torch.all(logical_to_rank_dispatch_physical_map != -1)

    # 获取原始张量的设备信息
    device = logical_to_all_physical_map.device
    # 返回属于当前ep_rank的分派映射表，并移动到正确的设备上
    return logical_to_rank_dispatch_physical_map[ep_rank, :, :].to(device)


# TODO optimize performance (rewrite and/or run in separate process with overlap)
def compute_logical_to_rank_dispatch_physical_map(
    logical_to_all_physical_map: torch.Tensor,
    num_gpus: int,
    num_physical_experts: int,
    ep_rank: int,
    seed: int = 42,
):
    """为逻辑专家到物理专家的分派计算一个静态映射表，优先选择本地专家。

    此函数为每个 (GPU, 逻辑专家) 对确定性地分配一个物理专家实例。其核心策略是最大化数据局部性，
    即优先将计算任务分配给与请求发起 GPU 位于同一计算节点上的专家（本地专家），以减少跨节点通信开销。

    1.  **本地优先分配 (Local-First)**：对于每个 GPU，函数会首先检查逻辑专家是否有任何物理副本位于该 GPU 上。
        如果存在本地副本，将从中选择一个进行任务分配。
    2.  **公平的远程分配 (Fair Remote Assignment)**：如果逻辑专家在当前 GPU 上没有可用的物理副本，
        函数将从位于其他 GPU 上的副本中进行选择。为了确保负载均衡，它会采用一种公平选择算法（fair_choices），
        在所有可用的远程专家中轮流分配，避免将过多请求集中于少数几个专家。
    3.  **确定性 (Determinism)**：整个分配过程是完全确定性的。通过使用固定的随机种子，可以保证每次调用
        生成的映射表都是相同的，这对于可复现性和系统稳定性至关重要。

    Args:
        logical_to_all_physical_map (torch.Tensor): 一个三维张量，映射每个逻辑专家到其所有物理副本。
            形状为 `(num_layers, num_logical_experts, num_replicas)`。
        num_gpus (int): 专家并行组中的 GPU 总数。
        num_physical_experts (int): 物理专家的总数。
        ep_rank (int): 当前进程在专家并行组中的排名。
        seed (int): 用于随机数生成器的种子，以确保确定性行为。

    Returns:
        torch.Tensor: 一个二维张量，为当前的 `ep_rank` 映射每个逻辑专家到一个物理专家。
                      形状为 `(num_layers, num_logical_experts)`。
    """
    r = random.Random(seed)

    # 计算每个GPU上的物理专家数量
    num_local_physical_experts = num_physical_experts // num_gpus
    # 获取映射表的维度信息
    num_layers, num_logical_experts, _ = logical_to_all_physical_map.shape
    dtype = logical_to_all_physical_map.dtype

    # 创建一个用于存储最终分派映射的张量，并用-1填充
    logical_to_rank_dispatch_physical_map = torch.full(
        size=(num_gpus, num_layers, num_logical_experts),
        fill_value=-1,
        dtype=dtype,
    )

    # 遍历每一层
    for layer_id in range(num_layers):
        # 遍历该层中的每一个逻辑专家
        for logical_expert_id in range(num_logical_experts):
            # 获取当前逻辑专家的所有物理副本ID
            candidate_physical_expert_ids = _logical_to_all_physical_raw(
                logical_to_all_physical_map, layer_id, logical_expert_id
            )
            # 获取当前逻辑专家在所有GPU上的分派映射视图
            output_partial = logical_to_rank_dispatch_physical_map[
                :, layer_id, logical_expert_id
            ]

            # --- 本地优先分配阶段 ---
            # 遍历所有GPU，尝试为每个GPU分配一个本地专家
            for gpu_id in range(num_gpus):
                # 找出位于当前GPU上的所有物理专家副本
                same_gpu_physical_expert_ids = [
                    physical_expert_id
                    for physical_expert_id in candidate_physical_expert_ids
                    if _compute_gpu_id_of_physical_expert(
                        physical_expert_id, num_local_physical_experts
                    )
                    == gpu_id
                ]
                # 如果找到了本地专家，则将第一个本地专家分配给当前GPU
                if len(same_gpu_physical_expert_ids) > 0:
                    output_partial[gpu_id] = same_gpu_physical_expert_ids[0]

            # --- 公平远程分配阶段 ---
            # 计算还有多少GPU没有被分配专家（即值为-1的条目）
            num_remain = torch.sum(output_partial == -1).item()
            # 为这些未分配的GPU，从所有候选物理专家中公平地选择并分配
            output_partial[output_partial == -1] = torch.tensor(
                _fair_choices(candidate_physical_expert_ids, k=num_remain, r=r),
                dtype=dtype,
            )

    # 断言确保所有条目都已被成功分配
    assert torch.all(logical_to_rank_dispatch_physical_map != -1)

    # 获取原始张量的设备信息
    device = logical_to_all_physical_map.device
    # 返回属于当前ep_rank的分派映射表，并移动到正确的设备上
    return logical_to_rank_dispatch_physical_map[ep_rank, :, :].to(device)


def _logical_to_all_physical_raw(
    logical_to_all_physical_map, layer_id: int, logical_expert_id: int
) -> List[int]:
    return [
        physical_expert_id
        for physical_expert_id in logical_to_all_physical_map[
            layer_id, logical_expert_id
        ].tolist()
        if physical_expert_id != -1
    ]


def _compute_gpu_id_of_physical_expert(
    physical_expert_id: int, num_local_physical_experts: int
) -> int:
    return physical_expert_id // num_local_physical_experts


def _fair_choices(arr: List, k: int, r: random.Random) -> List:
    quotient, remainder = divmod(k, len(arr))
    ans = arr * quotient + r.sample(arr, k=remainder)
    r.shuffle(ans)
    return ans


@dataclass
class ModelConfigForExpertLocation:
    num_layers: int
    num_logical_experts: int
    num_groups: Optional[int] = None

    @staticmethod
    def init_dummy():
        return ModelConfigForExpertLocation(num_layers=1, num_logical_experts=1)

    @staticmethod
    def from_model_config(model_config: ModelConfig):
        model_class, _ = get_model_architecture(model_config)
        if hasattr(model_class, "get_model_config_for_expert_location"):
            return model_class.get_model_config_for_expert_location(
                model_config.hf_config
            )
        else:
            return ModelConfigForExpertLocation.init_dummy()


def compute_initial_expert_location_metadata(
    server_args: ServerArgs, model_config: ModelConfig
) -> ExpertLocationMetadata:
    """
    根据服务配置计算并初始化专家位置元数据。

    这个函数作为 `ExpertLocationMetadata` 的工厂函数，根据 `server_args.init_expert_location`
    字段指定的策略来创建元数据。它支持多种初始化方法：

    1.  **"trivial"**: 创建一个简单的、一一对应的专家映射。适用于每个逻辑专家都直接映射到
        一个物理专家的场景。
    2.  **基于映射文件/字符串**: 从一个文件（.pt 或 .json）或 JSON 字符串中加载预定义的
        `physical_to_logical_map`。这允许用户提供一个精确的物理专家到逻辑专家的映射关系。
    3.  **基于 EPLB (Expert Placement via Load Balancing)**: 从一个包含 `logical_count`
        的文件/字符串中初始化。这通常用于需要根据专家使用频率或负载来动态分配专家的场景。

    Args:
        server_args (ServerArgs): 包含服务启动参数的对象，特别是 `init_expert_location`
                                  字段，它决定了初始化的方式。
        model_config (ModelConfig): 包含模型配置信息的对象，如层数、专家数等。

    Returns:
        ExpertLocationMetadata: 初始化完成的专家位置元数据对象。

    Raises:
        NotImplementedError: 如果 `init_expert_location` 指定的格式未知或不受支持。
    """
    data = server_args.init_expert_location
    if data == "trivial":
        # 使用简单的一一对应映射进行初始化
        return ExpertLocationMetadata.init_trivial(server_args, model_config)

    # 从文件或 JSON 字符串加载专家位置数据
    # TODO: 与工具函数统一加载逻辑
    if data.endswith(".pt"):
        data_dict = torch.load(data, weights_only=True)
    elif data.endswith(".json"):
        data_dict = json.loads(Path(data).read_text())
    else:
        data_dict = json.loads(data)

    if "physical_to_logical_map" in data_dict:
        # 根据预定义的物理到逻辑映射进行初始化
        logger.info("使用 ServerArgs.init_expert_location 从映射初始化专家位置")
        return ExpertLocationMetadata.init_by_mapping(
            server_args, model_config, **data_dict
        )
    elif "logical_count" in data_dict:
        # 根据 EPLB 的逻辑计数进行初始化
        logger.info("使用 ServerArgs.init_expert_location 从 EPLB 初始化专家位置")
        return ExpertLocationMetadata.init_by_eplb(
            server_args, model_config, logical_count=data_dict["logical_count"]
        )
    else:
        # 如果格式未知，则抛出错误
        raise NotImplementedError(
            f"未知的 init_expert_location 格式 ({list(data_dict.keys())=})"
        )
