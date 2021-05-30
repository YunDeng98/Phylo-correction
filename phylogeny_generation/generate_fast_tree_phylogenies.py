import argparse
import multiprocessing
import os
import sys

import hashlib
import logging
import numpy as np
import random
import tqdm

from phylogeny_generation.FastTreePhylogeny import FastTreePhylogeny


def init_logger():
    logger = logging.getLogger("phylogeny_generation")
    logger.setLevel(logging.DEBUG)
    # fmt_str = "[%(asctime)s] %(levelname)s @ line %(lineno)d: %(message)s"
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler("generate_fast_tree_phylogenies.log")
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)


parser = argparse.ArgumentParser(description="Run FastTree on all MSAs")
parser.add_argument(
    "--a3m_dir",
    type=str,
    help="Directory where the MSAs are found (.a3m files)",
    required=True,
)
parser.add_argument(
    "--outdir",
    type=str,
    help="Directory where the reconstructed phylogenies will be found.",
    required=True,
)
parser.add_argument(
    "--n_process",
    type=int,
    help="Number of processes to use",
    required=True,
)
parser.add_argument(
    "--expected_number_of_MSAs",
    type=int,
    help="Expected number of MSAs",
    required=True,
)
parser.add_argument(
    "--max_seqs",
    type=int,
    help="Maximum number of sequences to use per family",
    required=True,
)
parser.add_argument(
    "--max_sites",
    type=int,
    help="Maximum number of sites to use per family",
    required=True,
)
parser.add_argument(
    "--max_families",
    type=int,
    help="Maximum number of family to run on.",
    required=False,
    default=100000000
)
parser.add_argument(
    "--rate_matrix",
    type=str,
    help="Rate matrix to be used in FastTree.",
    required=True,
)


def _map_func(args):
    a3m_dir = args[0]
    protein_family_name = args[1]
    outdir = args[2]
    max_seqs = args[3]
    max_sites = args[4]
    rate_matrix = args[5]

    logger = logging.getLogger("phylogeny_generation")
    seed = int(hashlib.md5((protein_family_name + 'phylogeny_generation').encode()).hexdigest()[:8], 16)
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


class PhylogenyGenerator():
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
        rate_matrix: str
    ):
        self.a3m_dir = a3m_dir
        self.n_process = n_process
        self.expected_number_of_MSAs = expected_number_of_MSAs
        self.outdir = outdir
        self.max_seqs = max_seqs
        self.max_sites = max_sites
        self.max_families = max_families
        self.rate_matrix = rate_matrix

    def run(self):
        a3m_dir = self.a3m_dir
        n_process = self.n_process
        expected_number_of_MSAs = self.expected_number_of_MSAs
        outdir = self.outdir
        max_seqs = self.max_seqs
        max_sites = self.max_sites
        max_families = self.max_families
        rate_matrix = self.rate_matrix

        init_logger()

        if os.path.exists(outdir):
            raise ValueError(
                f"outdir {outdir} already exists. Aborting not to " f"overwrite!"
            )
        os.makedirs(outdir)

        if not os.path.exists(a3m_dir):
            raise ValueError(f"Could not find a3m_dir {a3m_dir}")

        filenames = list(os.listdir(a3m_dir))
        if not len(filenames) == expected_number_of_MSAs:
            raise ValueError(
                f"Number of MSAs is {len(filenames)}, does not match "
                f"expected {expected_number_of_MSAs}"
            )
        protein_family_names = [x.split(".")[0] for x in filenames][:max_families]

        map_args = [
            (a3m_dir, protein_family_name, outdir, max_seqs, max_sites, rate_matrix)
            for protein_family_name in protein_family_names
        ]
        with multiprocessing.Pool(n_process) as pool:
            list(tqdm.tqdm(pool.imap(_map_func, map_args), total=len(map_args)))


def _main():
    # Pull out arguments
    args = parser.parse_args()

    phylogeny_generator = PhylogenyGenerator(
        a3m_dir=args.a3m_dir,
        n_process=args.n_process,
        expected_number_of_MSAs=args.expected_number_of_MSAs,
        outdir=args.outdir,
        max_seqs=args.max_seqs,
        max_sites=args.max_sites,
        max_families=args.max_families,
        rate_matrix=args.rate_matrix,
    )

    phylogeny_generator.run()


if __name__ == "__main__":
    _main()
