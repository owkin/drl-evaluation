"""Test tracking sytem."""
import re
import shutil
from pathlib import Path

import pytest

from omics_rpz.exceptions import UnCommittedFilesError
from omics_rpz.experiment_tracking import git_interface


@pytest.fixture
def untracked_folder():
    """Fixture to create a tmp file not tracked by git."""
    new_file = Path(__file__).parent / "tmp" / "tmp" / "untracked.txt"
    new_file.parent.mkdir(parents=True, exist_ok=True)
    new_file.write_text("This is an untracked file")

    assert new_file.exists()

    untracked_folder = new_file.parents[1]

    repo_relative_path = re.split(r'omics(-|_)rpz/', str(untracked_folder))[-1]

    yield repo_relative_path

    shutil.rmtree(untracked_folder)


@pytest.fixture(autouse=True)
def unadded_change():
    """Modifyed file but not added."""
    tracked_file = Path(__file__).parent / "assets" / "tracked_file.txt"
    true_content = tracked_file.read_text()
    tracked_file.write_text('Changes')

    # should be tests/experiment_tracking/assets/tracked_file.txt
    repo_relative_path = re.split(r'omics(-|_)rpz/', str(tracked_file.parent))[-1]

    yield repo_relative_path

    tracked_file.write_text(true_content)


@pytest.fixture(autouse=True)  # Needs to be re executed at each call
def added_change(unadded_change):
    """Modifyed file but not added."""
    git_interface.REPO.git.add(unadded_change)

    yield unadded_change

    git_interface.REPO.git.restore(unadded_change, "--staged")


@pytest.mark.parametrize("path", ["untracked_folder", "unadded_change", "added_change"])
def test_untracked_file(path, request):
    """Checks if an untracked file is found."""
    with pytest.raises(UnCommittedFilesError):
        git_interface.are_tracked(path=request.getfixturevalue(path))


@pytest.mark.parametrize(
    "tag,exists",
    [("fake", False), ("experiment-4aacb5b827f9791c71b61a0e8315e0d7e4c9e2f0", True)],
)
def test_tag_already_exists(tag, exists):
    """Checks if git interface fetch existing tags properly."""
    assert exists == git_interface.tag_already_exists(tag=tag)
