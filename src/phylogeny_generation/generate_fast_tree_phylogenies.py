import multiprocessing
import os

import hashlib
import logging
import numpy as np
import random
import tqdm

from .FastTreePhylogeny import FastTreePhylogeny


def _map_func(args) -> None:
    a3m_dir = args[0]
    protein_family_name = args[1]
    outdir = args[2]
    max_seqs = args[3]
    max_sites = args[4]
    rate_matrix = args[5]

    logger = logging.getLogger("phylogeny_generation")
    seed = int(hashlib.md5((protein_family_name + "phylogeny_generation").encode()).hexdigest()[:8], 16)
    logger.info(f"Setting random seed to: {seed}")
    np.random.seed(seed)
    random.seed(seed)

    FastTreePhylogeny(
        a3m_dir=a3m_dir,
        protein_family_name=protein_family_name,
        outdir=outdir,
        max_seqs=max_seqs,
        max_sites=max_sites,
        rate_matrix=rate_matrix,
    )


class PhylogenyGenerator:
    r"""
    Given a directory with MSAs, generates a directory with one tree for each MSA.
    """

    def __init__(
        self,
        a3m_dir: str,
        n_process: int,
        expected_number_of_MSAs: int,
        outdir: str,
        max_seqs: int,
        max_sites: int,
        max_families: int,
        rate_matrix: str,
    ):
        self.a3m_dir = a3m_dir
        self.n_process = n_process
        self.expected_number_of_MSAs = expected_number_of_MSAs
        self.outdir = outdir
        self.max_seqs = max_seqs
        self.max_sites = max_sites
        self.max_families = max_families
        self.rate_matrix = rate_matrix

    def run(self) -> None:
        a3m_dir = self.a3m_dir
        n_process = self.n_process
        expected_number_of_MSAs = self.expected_number_of_MSAs
        outdir = self.outdir
        max_seqs = self.max_seqs
        max_sites = self.max_sites
        max_families = self.max_families
        rate_matrix = self.rate_matrix

        if os.path.exists(outdir):
            raise ValueError(f"outdir {outdir} already exists. Aborting not to " f"overwrite!")
        os.makedirs(outdir)

        if not os.path.exists(a3m_dir):
            raise ValueError(f"Could not find a3m_dir {a3m_dir}")

        filenames = list(os.listdir(a3m_dir))
        if not len(filenames) == expected_number_of_MSAs:
            raise ValueError(
                f"Number of MSAs is {len(filenames)}, does not match " f"expected {expected_number_of_MSAs}"
            )
        protein_family_names = [x.split(".")[0] for x in filenames][:max_families]

        map_args = [
            [a3m_dir, protein_family_name, outdir, max_seqs, max_sites, rate_matrix]
            for protein_family_name in protein_family_names
        ]
        with multiprocessing.Pool(n_process) as pool:
            list(tqdm.tqdm(pool.imap(_map_func, map_args), total=len(map_args)))
