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
    use_cached = args[6]

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
        use_cached=use_cached,
    )


class PhylogenyGenerator:
    r"""
    Given a directory with MSAs, generates a directory with one tree for each MSA.
    The hyperparameters of the PhylogenyGenerator object are provided in '__init__',
    and the PhylogenyGenerator is run only when the 'run' method is called.

    Args:
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
        max_families: Only estimate trees for the first 'max_families' files in a3m_dir.
            This is useful for testing and to see what happens if less data is used.
        rate_matrix: Path to the rate matrix to use within FastTree. If ends in 'None', then
            the default rate matrix will be used in FastTree.
        use_cached: If True and the output file already exists, FastTree will NOT be run.
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
        use_cached: bool = False,
    ):
        self.a3m_dir = a3m_dir
        self.n_process = n_process
        self.expected_number_of_MSAs = expected_number_of_MSAs
        self.outdir = outdir
        self.max_seqs = max_seqs
        self.max_sites = max_sites
        self.max_families = max_families
        self.rate_matrix = rate_matrix
        self.use_cached = use_cached

    def run(self) -> None:
        a3m_dir = self.a3m_dir
        n_process = self.n_process
        expected_number_of_MSAs = self.expected_number_of_MSAs
        outdir = self.outdir
        max_seqs = self.max_seqs
        max_sites = self.max_sites
        max_families = self.max_families
        rate_matrix = self.rate_matrix
        use_cached = self.use_cached

        if os.path.exists(outdir) and not use_cached:
            raise ValueError(f"outdir {outdir} already exists. Aborting not to " f"overwrite!")
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if not os.path.exists(a3m_dir):
            raise ValueError(f"Could not find a3m_dir {a3m_dir}")

        filenames = list(os.listdir(a3m_dir))
        if not len(filenames) == expected_number_of_MSAs:
            raise ValueError(
                f"Number of MSAs at {a3m_dir} is {len(filenames)}, does not match " f"expected {expected_number_of_MSAs}"
            )
        protein_family_names = [x.split(".")[0] for x in filenames][:max_families]

        map_args = [
            [a3m_dir, protein_family_name, outdir, max_seqs, max_sites, rate_matrix, use_cached]
            for protein_family_name in protein_family_names
        ]
        with multiprocessing.Pool(n_process) as pool:
            list(tqdm.tqdm(pool.imap(_map_func, map_args), total=len(map_args)))

        os.system(f"chmod -R 555 {outdir}")
