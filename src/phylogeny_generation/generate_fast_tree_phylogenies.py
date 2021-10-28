import multiprocessing
import os

import hashlib
import logging
import numpy as np
import random
import tqdm

from .FastTreePhylogeny import FastTreePhylogeny
from src.utils import subsample_protein_families


def map_func(args) -> None:
    a3m_dir = args[0]
    protein_family_name = args[1]
    outdir = args[2]
    max_seqs = args[3]
    max_sites = args[4]
    rate_matrix = args[5]
    use_cached = args[6]
    fast_tree_cats = args[7]

    logger = logging.getLogger("phylo_correction.phylogeny_generation")
    seed = int(hashlib.md5((protein_family_name + "phylogeny_generation").encode()).hexdigest()[:8], 16)
    # logger.info(f"Setting random seed to: {seed}")
    np.random.seed(seed)
    random.seed(seed)

    FastTreePhylogeny(
        a3m_dir=a3m_dir,
        protein_family_name=protein_family_name,
        outdir=outdir,
        max_seqs=max_seqs,
        max_sites=max_sites,
        rate_matrix=rate_matrix,
        fast_tree_cats=fast_tree_cats,
        use_cached=use_cached,
    )


class PhylogenyGenerator:
    r"""
    Given a directory with MSAs, generates a directory with one tree for each MSA.
    The hyperparameters of the PhylogenyGenerator object are provided in '__init__',
    and the PhylogenyGenerator is run only when the 'run' method is called.

    Args:
        a3m_dir_full: Directory with MSAs for ALL protein families. Used
            to determine which max_families will get subsampled.
        a3m_dir: Directory where the MSA files are found.
        n_process: Number of processes used to parallelize computation.
        expected_number_of_MSAs: The number of files in a3m_dir. This argument
            is only used to sanity check that the correct a3m_dir is being used.
            It has no functional implications.
        outdir: Directory where the trees will be written out to (.newick files).
        max_seqs: If nonzero, this number of sequences in the MSA files will be subsampled
            uniformly at random. The first sequence in the MSA files will always be sampled.
        max_sites: If nonzero, this number of sites in the MSA files will be subsampled
            uniformly at random.
        max_families: Only estimate trees for 'max_families' randomly chosen files in a3m_dir.
            This is useful for testing and to see what happens if less data is used.
        rate_matrix: Path to the rate matrix to use within FastTree. If ends in 'None', then
            the default rate matrix will be used in FastTree.
        use_cached: If True and the output file already exists, FastTree will NOT be run.
    """

    def __init__(
        self,
        a3m_dir_full: str,
        a3m_dir: str,
        n_process: int,
        expected_number_of_MSAs: int,
        outdir: str,
        max_seqs: int,
        max_sites: int,
        max_families: int,
        rate_matrix: str,
        fast_tree_cats: int,
        use_cached: bool = False,
    ):
        self.a3m_dir_full = a3m_dir_full
        self.a3m_dir = a3m_dir
        self.n_process = n_process
        self.expected_number_of_MSAs = expected_number_of_MSAs
        self.outdir = outdir
        self.max_seqs = max_seqs
        self.max_sites = max_sites
        self.max_families = max_families
        self.rate_matrix = rate_matrix
        self.fast_tree_cats = fast_tree_cats
        self.use_cached = use_cached

    def run(self) -> None:
        logger = logging.getLogger("phylo_correction.generate_fast_tree_phylogenies")
        logger.info(f"Starting on max_families={self.max_families}, outdir: {self.outdir}")

        a3m_dir_full = self.a3m_dir_full
        a3m_dir = self.a3m_dir
        n_process = self.n_process
        expected_number_of_MSAs = self.expected_number_of_MSAs
        outdir = self.outdir
        max_seqs = self.max_seqs
        max_sites = self.max_sites
        max_families = self.max_families
        rate_matrix = self.rate_matrix
        fast_tree_cats = self.fast_tree_cats
        use_cached = self.use_cached

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if not os.path.exists(a3m_dir):
            raise ValueError(f"Could not find a3m_dir {a3m_dir}")

        protein_family_names = subsample_protein_families(
            a3m_dir_full,
            expected_number_of_MSAs,
            max_families
        )

        map_args = [
            [a3m_dir, protein_family_name, outdir, max_seqs, max_sites, rate_matrix, use_cached, fast_tree_cats]
            for protein_family_name in protein_family_names
        ]
        if n_process > 1:
            with multiprocessing.Pool(n_process) as pool:
                list(tqdm.tqdm(pool.imap(map_func, map_args), total=len(map_args)))
        else:
            list(tqdm.tqdm(map(map_func, map_args), total=len(map_args)))
