"""Exceptions for the package."""


class UnCommittedFilesError(Exception):
    """Raises if files are not committed while tracking an experiment."""


class UnPushedCommit(Exception):
    """Raises if commit is not pushed while tracking an experiment."""


class ResultFolderNotFound(Exception):
    """Raises if a non existing folder is trying to be uploaded to the results
    storage."""
