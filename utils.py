import json
import logging
import os

import git

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def git_log(path, str):
    repo = git.Repo(search_parent_directories=True)
    repo_info={
        "repo_id": str(repo),
        "repo_sha": str(repo.head.object.hexsha),
        "repo_branch": str(repo.active_branch)
    }

    with open(os.path.join(path, "git_log.json"), "w") as f:
        json.dump(repo_info, f, indent=4)


class Iters:
    def __init__(self, args_dict):
        for k, v in args_dict.items():
            setattr(self, k, v)
