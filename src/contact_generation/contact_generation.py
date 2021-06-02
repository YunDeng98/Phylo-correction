import logging
import multiprocessing
import os
import tqdm
import hashlib
import numpy as np
import random

from typing import List

from src.contact_generation.ContactMatrix import ContactMatrix


def map_func(args: List) -> None:
    pdb_dir = args[0]
    protein_family_name = args[1]
    outdir = args[2]
    armstrong_cutoff = args[3]
    use_cached = args[4]

    logger = logging.getLogger("contact_generation")

    # Caching pattern: skip any computation as soon as possible
    outfile = os.path.join(outdir, protein_family_name + ".cm")
    if use_cached and os.path.exists(outfile):
        logger.info(f"Skipping. Cached contact matrix for family {protein_family_name} at {outfile}")
        return

    seed = int(hashlib.md5((protein_family_name + "contact_generation").encode()).hexdigest()[:8], 16)
    logger.info(f"Setting random seed to: {seed}")
    np.random.seed(seed)
    random.seed(seed)

    contact_matrix = ContactMatrix(
        pdb_dir=pdb_dir,
        protein_family_name=protein_family_name,
        armstrong_cutoff=armstrong_cutoff,
    )
    outfile = os.path.join(outdir, protein_family_name + ".cm")
    contact_matrix.write_to_file(outfile)


class ContactGenerator:
    def __init__(
        self,
        a3m_dir: str,
        pdb_dir: str,
        armstrong_cutoff: float,
        n_process: int,
        expected_number_of_families: int,
        outdir: str,
        max_families: int,
        use_cached: bool = False,
    ):
        self.a3m_dir = a3m_dir
        self.pdb_dir = pdb_dir
        self.armstrong_cutoff = armstrong_cutoff
        self.n_process = n_process
        self.expected_number_of_families = expected_number_of_families
        self.outdir = outdir
        self.max_families = max_families
        self.use_cached = use_cached

    def run(self) -> None:
        a3m_dir = self.a3m_dir
        pdb_dir = self.pdb_dir
        armstrong_cutoff = self.armstrong_cutoff
        n_process = self.n_process
        expected_number_of_families = self.expected_number_of_families
        outdir = self.outdir
        max_families = self.max_families
        use_cached = self.use_cached

        if os.path.exists(outdir) and not use_cached:
            raise ValueError(f"outdir {outdir} already exists. Aborting not to " f"overwrite!")

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if not os.path.exists(pdb_dir):
            raise ValueError(f"Could not find pdb_dir {pdb_dir}")

        if not os.path.exists(a3m_dir):
            raise ValueError(f"Could not find pdb_dir {a3m_dir}")

        filenames = list(os.listdir(a3m_dir))
        if not len(filenames) == expected_number_of_families:
            raise ValueError(
                f"Number of families is {len(filenames)}, does not match " f"expected {expected_number_of_families}"
            )
        protein_family_names = [x.split(".")[0] for x in filenames][:max_families]

        map_args = [
            [pdb_dir, protein_family_name, outdir, armstrong_cutoff, use_cached]
            for protein_family_name in protein_family_names
        ]
        if n_process > 1:
            with multiprocessing.Pool(n_process) as pool:
                list(tqdm.tqdm(pool.imap(map_func, map_args), total=len(map_args)))
        else:
            list(tqdm.tqdm(map(map_func, map_args), total=len(map_args)))

        os.system(f"chmod -R 555 {outdir}")
