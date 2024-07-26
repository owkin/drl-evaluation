"""Functions to interact with git from python."""
import re
from pathlib import Path

from git import Repo
from loguru import logger
from omegaconf import DictConfig

from omics_rpz import exceptions

REPO_FOLDER = Path(__file__).parents[2]
REPO = Repo(REPO_FOLDER)


def are_tracked(path: str = None):
    """Checks wether or not there is untracked file per git within the given path.

    Args:
        path (str, optional): Subfolder of a the running git directory.
            If set to None, the repo root directory is used. Defaults to None.

    Raises:
        exceptions.UnCommittedFilesError: Files are not commited
    """
    if (
        REPO.is_dirty(path=path)
        or len([f for f in REPO.untracked_files if f.startswith(path or "")]) > 0
    ):
        raise exceptions.UnCommittedFilesError(
            "You are trying to track an experiment but some files are "
            "not commit. \nYou can use the ``git status`` command to see them. "
            "Please add them and commit them before "
            "launching  a new experiment or set the ``track`` parameter to false "
            "in hydra config conf/config.yaml"
        )


def is_pushed() -> str:
    """Checks if the latest commit has been pushed to remote.

    Raises:
        exceptions.UnPushedCommit: Local commit is not pushed.
    """
    error_msg = (
        "You are trying to track an experiment but the running commit is not "
        "pushed. Please push your changes or set the ``track`` parameter to false "
        "in hydra config conf/config.yaml"
    )
    remote = REPO.remote("origin")
    remote.fetch()

    if REPO.git.branch("-r", "--contains", REPO.head.commit) == "":
        raise exceptions.UnPushedCommit(error_msg)


def store_reference(tag: str):
    """Store running commit as a tag in origin. The commit will be tagged:
    ``experiment-<commit>``.

    Args:
        tag(str): Provided tag to be pushed
    """
    if tag_already_exists(tag):
        logger.info("The experiment code version has already been uploaded.")

    else:
        logger.info(f"Storing the code version as the git tag: {tag}")
        REPO.git.tag(tag)
        REPO.git.push('origin', tag)

    return tag


def experiment_id(cfg: DictConfig) -> str:
    """Checks wether the experiment can be tracked.

    Args:
        cfg (DictConfig): Hydra config from experiment

    Returns:
        str: Id for the experiment.
    """

    if cfg["track"]:
        are_tracked(path=None)
        is_pushed()
        tag = "experiment-" + str(REPO.head.commit)

    else:
        tag = None

    return tag


def tag_already_exists(tag: str) -> bool:
    """Checks wether or not the reference of the experiment already exists.

    Args:
        tag (str): Git taf of the experiment

    Returns:
        bool: If the reference already exists.
    """

    REPO.git.fetch('--all', '--tags')
    all_tags = getattr(REPO.git, "ls-remote")("--tags", "origin")
    all_tags = re.split(r'\t|\n', all_tags)
    exists = any(t.endswith("/tags/" + tag) for t in all_tags)

    return exists
