import os
import random
import hashlib

from typing import List


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
            f"match " f"expected {expected_number_of_MSAs}"
        )
    protein_family_names = [x.split(".")[0] for x in filenames][:max_families]
    return protein_family_names


def hash_str(a_string: str):
    return hashlib.sha512(a_string.encode()).hexdigest()[:8]
