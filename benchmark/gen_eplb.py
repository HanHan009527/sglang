import sys
import os
import json
from sglang.srt.server_args import prepare_server_args
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.eplb.expert_location import ExpertLocationMetadata
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, world_size, server_args):
    """The worker function for each process."""
    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'  # Use a free port

    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model_config = ModelConfig.from_server_args(server_args)

    expert_location = ExpertLocationMetadata.init_trivial(server_args, model_config)

    print(f"--- Rank {rank} Result ---")
    print(expert_location)
    print("-" * 20)

    # Export the object to disk
    output_dir = "/tmp/expert_location_metadata"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"expert_metadata_rank_{rank}.json")
    data_to_save = {
        "physical_to_logical_map": expert_location.physical_to_logical_map.cpu()
        .numpy()
        .tolist(),
        "logical_to_all_physical_map": expert_location.logical_to_all_physical_map.cpu()
        .numpy()
        .tolist(),
        "logical_to_all_physical_map_num_valid": expert_location.logical_to_all_physical_map_num_valid.cpu()
        .numpy()
        .tolist(),
    }
    if expert_location.logical_to_rank_dispatch_physical_map is not None:
        data_to_save[
            "logical_to_rank_dispatch_physical_map"
        ] = expert_location.logical_to_rank_dispatch_physical_map.cpu().numpy().tolist()

    with open(file_path, "w") as f:
        json.dump(data_to_save, f, indent=4)
    
    print(f"Saved expert location metadata for rank {rank} to {file_path}")

    # Clean up the process group
    dist.destroy_process_group()

def main():
    server_args = prepare_server_args(sys.argv[1:])
    world_size = server_args.ep_size

    mp.spawn(worker,
             args=(world_size, server_args),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()