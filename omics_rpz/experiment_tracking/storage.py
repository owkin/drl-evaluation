"""Interaction with the tracking experiment storage."""

import glob
import re
from os import PathLike
from pathlib import Path
from typing import Optional

import google.cloud.storage as gcp_storage
from loguru import logger

from omics_rpz.exceptions import ResultFolderNotFound


def list_local_files(path: Path) -> list[Path]:
    """List all files from local folder. If a file path is given it is return within a
    list.

    Args:
        path (Path): Folder to list files from

    Returns:
        list[Path]: list of absolute pathlib Path.
    """

    if path.is_file():
        return [path.resolve()]

    paths = glob.glob(str(path / "*"))
    files = [f for f in paths if Path(f).is_file()]

    return files


def file_to_blob(
    file: str,
    bucket: gcp_storage.Bucket,
    local_prefix: Optional[str] = None,
    remote_prefix: str = "results",
) -> gcp_storage.Blob:
    """Converts file path to a GCP Blob object associated to the given bucket. The
    remote prefix will replace the local prefix specified.

    Example:

        ``/Users/owkin/myfile.txt`` will be converted to ``results/myfile.txt``
        if the given ``prefix`` is ``/Users/owkin/myfile.txt``

    Args:
        file (str): Absolute path to a local file.
        bucket (gcp_storage.Bucket): GCP local bucket where the file will be uploaded.
        local_prefix (Optional[str], optional): Prefix to remove from local file path.
            If set to None, no prefix will be removed. Defaults to None.
        remote_prefix (str, optional): Prefix to add to the path to be uploaded.
            Defaults to "results".

    Returns:
        gcp_storage.Blob: GCP Blob pointing to the given bucket.
    """
    if local_prefix is not None:
        file = re.sub(rf"^{local_prefix}", "", file)

    file = re.sub(r"^/", "", file)
    remote_file_path = str(remote_prefix / Path(file))
    blob = bucket.blob(remote_file_path)

    return blob


def upload_folder(path: PathLike, bucket: str, remote_prefix: str = "results"):
    """Upload en entire folder to the given bucket. The uploaded folder path will be:
    ``<remote_prefix>/path.stem/...``.

    Args:
        path (PathLike): Local folder to upload.
        bucket (str): Bucket name where to upload the folder.
        remote_prefix (str, optional): Prefix to add to the folder path.
            Defaults to "results".

    Raises:
        ResultFolderNotFound: If the folder doesn't exist.
    """

    if not Path(path).exists():
        raise ResultFolderNotFound("The given path ``{path}`` do not exists.")

    gcp_client = gcp_storage.Client()
    bucket = gcp_client.bucket(bucket)

    absolute_path = Path(path).resolve()

    files_to_upload = list_local_files(absolute_path)
    files_and_blobs = [
        (
            f,
            file_to_blob(
                f,
                bucket=bucket,
                local_prefix=str(absolute_path.parent),
                remote_prefix=remote_prefix,
            ),
        )
        for f in files_to_upload
    ]
    logger.info(f"Uploading experiment results to GCP bucket ``{bucket}``")

    for file, blob in files_and_blobs:
        blob.upload_from_filename(file)
