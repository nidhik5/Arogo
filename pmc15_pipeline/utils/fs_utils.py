import functools
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


@functools.cache
def get_repo_root_path(known_root_foldername=".vscode"):
    for parent in Path(__file__).parents:
        if (parent / known_root_foldername).exists():
            return parent

    # check we're not at the root of the drive `/`
    raise ValueError(
        f"Repo root could not be found! Did not find `{known_root_foldername}` as child of any folders in path {Path(__file__)}"
    )


def get_line_count(file_path: Path) -> int:
    return sum(1 for _ in open(file_path, "r", encoding="utf8"))
