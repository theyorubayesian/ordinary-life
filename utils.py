import json
import logging
import os
import random
import socket

import git
import numpy as np
import torch

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(args):
    """
    Set the random seed.
    """
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def git_log(path, str):
    repo = git.Repo(search_parent_directories=True)
    repo_info={
        "repo_id": str(repo),
        "repo_sha": str(repo.head.object.hexsha),
        "repo_branch": str(repo.active_branch)
    }

    with open(os.path.join(path, "git_log.json"), "w") as f:
        json.dump(repo_info, f, indent=4)


def init_gpu_params(args):
    if args.n_gpu <= 0:
        args.local_rank = 0
        args.master_port = -1
        args.is_master = True,
        args.multi_gpu = False
    
    assert torch.cuda.is_available()

    logger.info("Initializing GPUs(s)")

    if args.n_gpu > 1:
        assert args.local_rank != -1

        args.world_size = int(os.getenv("WORLD_SIZE"))
        args.n_gpu_per_node = int(os.getenv("N_GPU_NODE"))
        args.global_rank = int(os.getenv("RANK"))

        args.n_nodes = args.world_size // args.n_gpu_per_node
        args.node_id = args.global_rank // args.n_gpu_per_node
        args.multi_gpu = True
    else:
        args.n_nodes = 1
        args.node_id = 0
        args.local_rank = 0 if args.local_rank == -1 else args.local_rank
        args.global_rank = 1
        args.world_size= 2 # TODO: How to check number of GPUs
        args.n_gpu_per_node = 2
        args.multi_gpu = False
        args.is_master = True
    
    assert args.n_nodes >= 1
    assert 0 <= args.node_id < args.n_nodes
    assert 0 <= args.local_rank <= args.global_rank < args.world_size
    assert args.world_size == args.n_nodes * args.n_gpu_per_node

    args.multi_node = args.n_nodes > 1

     # summary
    PREFIX = f"--- Global rank: {args.global_rank} - "
    logger.info(PREFIX + "Number of nodes: %i" % args.n_nodes)
    logger.info(PREFIX + "Node ID        : %i" % args.node_id)
    logger.info(PREFIX + "Local rank     : %i" % args.local_rank)
    logger.info(PREFIX + "World size     : %i" % args.world_size)
    logger.info(PREFIX + "GPUs per node  : %i" % args.n_gpu_per_node)
    logger.info(PREFIX + "Master         : %s" % str(args.is_master))
    logger.info(PREFIX + "Multi-node     : %s" % str(args.multi_node))
    logger.info(PREFIX + "Multi-GPU      : %s" % str(args.multi_gpu))
    logger.info(PREFIX + "Hostname       : %s" % socket.gethostname())

    torch.cuda.set_device(args.local_rank)

    if args.multi_gpu:
        torch.distributed.init_process_grup(
            init_method="env://",
            backend="nccl"
        )


class Iters:
    def __init__(self, args_dict):
        for k, v in args_dict.items():
            setattr(self, k, v)
