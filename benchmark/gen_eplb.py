import argparse
from sglang.srt.server_args import ServerArgs
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.eplb.expert_location import ExpertLocationMetadata

def main(args):
    server_args = ServerArgs.from_cli_args(args)
    model_config = ModelConfig.from_server_args(server_args)
    
    expert_location = ExpertLocationMetadata.init_trivial(server_args, model_config)
    
    print(expert_location)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)