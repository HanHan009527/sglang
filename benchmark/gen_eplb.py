import argparse
import os
from sglang.srt.server_args import ServerArgs
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.eplb.expert_location import ExpertLocationMetadata
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, world_size, args):
    """The worker function for each process."""
    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'  # Use a free port

    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    server_args = ServerArgs.from_cli_args(args)
    model_config = ModelConfig.from_server_args(server_args)

    expert_location = ExpertLocationMetadata.init_trivial(server_args, model_config)
    
    print(f"--- Rank {rank} Result ---")
    print(expert_location)
    print("-" * 20)

    # Clean up the process group
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    
    world_size = args.ep_size
    mp.spawn(worker,
             args=(world_size, args),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()