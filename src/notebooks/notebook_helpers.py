import os
from pathlib import Path
from typing import Optional

# Markers that identify the repository root. A directory is the root if it
# contains all of them.
_ROOT_MARKERS = ("src", "data", "experiments")


def find_repo_root(start: Optional[str] = None) -> Path:
    """
    Walk upwards from `start` (default: this file) until a directory holding
    every entry in _ROOT_MARKERS is found.

    Raises:
        RuntimeError: if no such directory exists between `start` and the
            filesystem root, which means the notebook is being run from outside
            a checkout of this repository.
    """
    here = Path(start).resolve() if start else Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if candidate.is_dir() and all(
            (candidate / marker).is_dir() for marker in _ROOT_MARKERS
        ):
            return candidate
    raise RuntimeError(
        f"Could not locate the repository root above {here}. Expected an "
        f"ancestor directory containing {list(_ROOT_MARKERS)}."
    )


def set_working_dir_to_root(root: Optional[str] = None) -> Path:
    """
    chdir to the repository root so the relative paths used throughout the
    codebase (./data/..., ./experiments/...) resolve from a notebook.

    Args:
        root: override the detected root. Leave unset in the normal case.

    Returns:
        The directory changed into.
    """
    target = Path(root).resolve() if root else find_repo_root()
    os.chdir(target)
    return target
