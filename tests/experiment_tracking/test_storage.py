"""Unitest the experiment tracking module."""


from pathlib import Path
from unittest.mock import MagicMock, Mock

import google.cloud.storage as gcp_storage
import pytest

from omics_rpz.experiment_tracking import storage


@pytest.fixture
def fake_bucket():
    """Fake bucket to be used to test function relying on a GCP Bucket object."""
    bucket = Mock(spec=gcp_storage.Bucket)
    bucket.blob = MagicMock(
        side_effect=lambda path: gcp_storage.Blob(name=path, bucket="fake-bucket")
    )

    return bucket


def test_list_local_files():
    """Checks that the list files function returns 2 when applied to the asset
    folder."""
    path = Path(__file__).parent / "assets"

    assert len(storage.list_local_files(path)) == 2


def test_list_local_file():
    """Checks that the list files function returns 2 when applied to the asset
    folder."""
    path = Path(__file__).parent / "assets" / "file_to_be_listed"

    assert len(storage.list_local_files(path)) == 1


@pytest.mark.parametrize(
    "file, prefix",
    [('/prefix/test.txt', "/prefix"), ("/test.txt", None), ("test.txt", None)],
)
def test_file_to_blob(fake_bucket, file, prefix):
    """Checks that the blob file name is rightly prefixed."""
    expected = 'results/test.txt'

    assert (
        storage.file_to_blob(
            file=file, local_prefix=prefix, bucket=fake_bucket, remote_prefix="results"
        ).name
        == expected
    )
