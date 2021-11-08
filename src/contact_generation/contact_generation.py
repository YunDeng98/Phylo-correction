import logging
import multiprocessing
import os
import tqdm
import hashlib
import numpy as np
import random

from typing import List

from src.contact_generation.ContactMatrix import ContactMatrix
from src.utils import subsample_protein_families, verify_integrity


def map_func(args: List) -> None:
    pdb_dir = args[0]
    protein_family_name = args[1]
    outdir = args[2]
    armstrong_cutoff = args[3]
    use_cached = args[4]

    logger = logging.getLogger("phylo_correction.contact_generation")

    # Caching pattern: skip any computation as soon as possible
    outfile = os.path.join(outdir, protein_family_name + ".cm")
    if use_cached and os.path.exists(outfile):
        verify_integrity(outfile)
        # logger.info(f"Skipping. Cached contact matrix for family {protein_family_name} at {outfile}")
        return

    seed = int(hashlib.md5((protein_family_name + "contact_generation").encode()).hexdigest()[:8], 16)
    # logger.info(f"Setting random seed to: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"Starting on family {protein_family_name}")

    contact_matrix = ContactMatrix(
        pdb_dir=pdb_dir,
        protein_family_name=protein_family_name,
        armstrong_cutoff=armstrong_cutoff,
    )
    outfile = os.path.join(outdir, protein_family_name + ".cm")
    contact_matrix.write_to_file(outfile)
    os.system(f"chmod 555 {outfile}")


class ContactGenerator:
    r"""
    Generates contact matrices from PDB files.

    All hyperparameters are provided upon '__init__', and the contact
    matrices are only generated when the 'run' method is called.

    Args:
        a3m_dir_full: Directory with MSAs for ALL protein families. Used
            to determine which max_families will get subsampled.
        a3m_dir: Directory where the MSA files (.a3m) are found. Although they
            are never read, this must be provided to be able to subsample families via the 'max_families' argument.
        pdb_dir: Directory where the PDB structure files (.pdb) are found.
        armstrong_cutoff: Armstrong cutoff threshold used to determine if two
            sites are in contact.
        n_process: Number of processes used to parallelize computation.
        expected_number_of_families: The number of files in a3m_dir. This argument
            is only used to sanity check that the correct a3m_dir is being used.
            It has no functional implications.
        outdir: Directory where the generated contact matrices will be found (.cm files)
        max_families: Only run on 'max_families' randomly chosen files in a3m_dir.
            This is useful for testing and to see what happens if less data is used.
        use_cached: If True and the output file already exists for a family,
            all computation will be skipped for that family.
    """
    def __init__(
        self,
        a3m_dir_full: str,
        a3m_dir: str,
        pdb_dir: str,
        armstrong_cutoff: float,
        n_process: int,
        expected_number_of_families: int,
        outdir: str,
        max_families: int,
        use_cached: bool = False,
    ):
        self.a3m_dir_full = a3m_dir_full
        self.a3m_dir = a3m_dir
        self.pdb_dir = pdb_dir
        self.armstrong_cutoff = armstrong_cutoff
        self.n_process = n_process
        self.expected_number_of_families = expected_number_of_families
        self.outdir = outdir
        self.max_families = max_families
        self.use_cached = use_cached

    def run(self) -> None:
        logger = logging.getLogger("phylo_correction.contact_generation")
        logger.info(f"Starting on max_families={self.max_families}, outdir: {self.outdir}")

        a3m_dir_full = self.a3m_dir_full
        a3m_dir = self.a3m_dir
        pdb_dir = self.pdb_dir
        armstrong_cutoff = self.armstrong_cutoff
        n_process = self.n_process
        expected_number_of_families = self.expected_number_of_families
        outdir = self.outdir
        max_families = self.max_families
        use_cached = self.use_cached

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if not os.path.exists(pdb_dir):
            raise ValueError(f"Could not find pdb_dir {pdb_dir}")

        if not os.path.exists(a3m_dir):
            raise ValueError(f"Could not find pdb_dir {a3m_dir}")

        protein_family_names = subsample_protein_families(
            a3m_dir_full,
            expected_number_of_families,
            max_families
        )

        map_args = [
            [pdb_dir, protein_family_name, outdir, armstrong_cutoff, use_cached]
            for protein_family_name in protein_family_names
        ]
        if n_process > 1:
            with multiprocessing.Pool(n_process) as pool:
                list(tqdm.tqdm(pool.imap(map_func, map_args), total=len(map_args)))
        else:
            list(tqdm.tqdm(map(map_func, map_args), total=len(map_args)))
