import contextlib
import hashlib
import logging
import os
import random
from typing import List

logger = logging.getLogger("phylo_correction.utils")


@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


def subsample_protein_families(
    a3m_dir_full: str,
    expected_number_of_MSAs: int,
    max_families: int,
) -> List[str]:
    if not os.path.exists(a3m_dir_full):
        raise ValueError(f"Could not find a3m_dir_full {a3m_dir_full}")

    filenames = sorted(list(os.listdir(a3m_dir_full)))
    random.Random(123).shuffle(filenames)
    if not len(filenames) == expected_number_of_MSAs:
        raise ValueError(
            f"Number of MSAs at {a3m_dir_full} is {len(filenames)}, does not "
            f"match "
            f"expected {expected_number_of_MSAs}"
        )
    protein_family_names = [x.split(".")[0] for x in filenames][:max_families]
    return protein_family_names


def hash_str(a_string: str):
    return hashlib.sha512(a_string.encode()).hexdigest()[:8]


def verify_integrity(filepath: str, mode: str):
    if not os.path.exists(filepath):
        logger.error(
            f"Trying to verify the integrity of an inexistent file: {filepath}"
        )
        raise Exception(
            f"Trying to verify the integrity of an inexistent file: {filepath}"
        )
    mask = oct(os.stat(filepath).st_mode)[-3:]
    if mask != mode:
        logger.error(
            f"filename {filepath} does not have status {mode}. Instead, it "
            f"has status: {mask}. It is most likely corrupted."
        )
        raise Exception(
            f"filename {filepath} does not have status {mode}. Instead, it "
            f"has status: {mask}. It is most likely corrupted."
        )


def verify_integrity_of_directory(
    dirpath: str, expected_number_of_files: int, mode: str = "555"
):
    """
    Makes sure that the directory has the expected number of files and that
    they are all write protected (or another specified mode).
    """
    dirpath = os.path.abspath(dirpath)
    if not os.path.exists(dirpath):
        logger.error(
            f"Trying to verify the integrity of an inexistent "
            f"directory: {dirpath}"
        )
        raise Exception(
            f"Trying to verify the integrity of an inexistent "
            f"diretory: {dirpath}"
        )
    filenames = sorted(list(os.listdir(dirpath)))
    if len(filenames) != expected_number_of_files:
        raise Exception(
            f"{dirpath} already exists but does not contain the "
            "expected_number_of_files."
            f"\nExpected: {expected_number_of_files}\nFound: {len(filenames)}"
        )
    for filename in filenames:
        filepath = os.path.join(dirpath, filename)
        verify_integrity(filepath=filepath, mode=mode)
