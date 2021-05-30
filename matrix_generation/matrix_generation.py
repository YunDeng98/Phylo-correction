r"""
Reads transition edges and summarizes them into transition matrices based on branch
length quantization.
"""
import argparse
import multiprocessing
import os
import sys

import logging
import numpy as np
import tempfile
import tqdm
from typing import Dict, List, Tuple

from ete3 import Tree
import numpy as np
import pandas as pd
import string
import random
import hashlib

sys.path.append('../')


def init_logger():
    logger = logging.getLogger("matrix_generation")
    logger.setLevel(logging.DEBUG)
    # fmt_str = "[%(asctime)s] %(levelname)s @ line %(lineno)d: %(message)s"
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler("matrix_generation.log")
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)


parser = argparse.ArgumentParser(description="Generate dataset of transitions based on phylogenies with ancestral states.")
parser.add_argument(
    "--a3m_dir",
    type=str,
    help="Directory where the MSAs are found (.a3m files)",
    required=True,
)
parser.add_argument(
    "--transitions_dir",
    type=str,
    help="Directory where the transitions are found (.transitions files)",
    required=True,
)
parser.add_argument(
    "--outdir",
    type=str,
    help="Directory where the matrices will be written to",
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
# parser.add_argument(
#     "--max_seqs",
#     type=int,
#     help="Maximum number of sequences to use per family",
#     required=True,
# )
# parser.add_argument(
#     "--max_sites",
#     type=int,
#     help="Maximum number of sites to use per family",
#     required=True,
# )
parser.add_argument(
    "--max_families",
    type=int,
    help="Maximum number of family to run on.",
    required=False,
    default=100000000
)
parser.add_argument(
    "--num_sites",
    type=int,
    help="Whether the transitions are single-site (1) or for two sites (2)",
    required=True
)


def map_func(args):
    a3m_dir = args[0]
    transitions_dir = args[1]
    protein_family_names_for_shard = args[2]
    outdir = args[3]
    alphabet = args[4]

    logger = logging.getLogger("matrix_generation")
    seed = int(hashlib.md5((''.join(protein_family_names_for_shard) + 'matrix_generation').encode()).hexdigest()[:8], 16)
    logger.info(f"Setting random seed to: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"Starting on {len(protein_family_names_for_shard)} families")

    # Create results data frame.
    res = pd.DataFrame(np.zeros(shape=(len(alphabet), len(alphabet)), dtype=int), index=alphabet, columns=alphabet)

    for protein_family_name in protein_family_names_for_shard:
        print(f"Starting on {protein_family_name}")
        transitions_df = pd.read_csv(os.path.join(transitions_dir, protein_family_name + ".transitions"), sep=",")

        # Filter transitions based on citeria
        print(f"TODO: Filter transitions based on criteria! (low, short branches)")

        # Now add quantization column and group by it too.
        print(f"TODO: Add quantization column and group by it too!")

        # Summarize remaining transitions into the matrices
        summarized_transitions = transitions_df.groupby(["starting_state", "ending_state"]).size()
        for starting_state in alphabet:
            for ending_state in alphabet:
                if (starting_state, ending_state) in summarized_transitions:
                    res.loc[starting_state, ending_state] += summarized_transitions[(starting_state, ending_state)]

    return res


def write_out_matrices(res, outdir):
    out_filepath = os.path.join(outdir, "matrices.txt")
    res.to_csv(out_filepath, sep="\t")


def get_protein_family_names_for_shard(
    shard_id: int,
    n_process: int,
    protein_family_names: List[str]
) -> List[str]:
    res = [protein_family_names[i] for i in range(len(protein_family_names)) if i % n_process == shard_id]
    return res


class MatrixGenerator:
    def __init__(
        self,
        a3m_dir,
        transitions_dir,
        n_process,
        expected_number_of_MSAs,
        outdir,
        max_families,
        num_sites,
    ):
        self.a3m_dir = a3m_dir
        self.transitions_dir = transitions_dir
        self.n_process = n_process
        self.expected_number_of_MSAs = expected_number_of_MSAs
        self.outdir = outdir
        self.max_families = max_families
        self.num_sites = num_sites

    def run(self):
        a3m_dir = self.a3m_dir
        transitions_dir = self.transitions_dir
        n_process = self.n_process
        expected_number_of_MSAs = self.expected_number_of_MSAs
        outdir = self.outdir
        max_families = self.max_families
        num_sites = self.num_sites

        assert(num_sites in [1, 2])

        # Create list of amino acids
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'] + ['-']
        for letter in string.ascii_uppercase:
            if letter not in amino_acids:
                amino_acids += [letter]

        # Create alphabet of states.
        if num_sites == 2:
            alphabet = []
            for aa1 in amino_acids:
                for aa2 in amino_acids:
                    alphabet.append(f"{aa1}{aa2}")
        else:
            assert(num_sites == 1)
            alphabet = amino_acids[:]

        init_logger()
        logger = logging.getLogger("matrix_generation")
        logger.info("Starting ... ")

        print(f"TODO: Accept branch length quantization as input!")

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
            (a3m_dir, transitions_dir, get_protein_family_names_for_shard(shard_id, n_process, protein_family_names), outdir, alphabet)
            for shard_id in range(n_process)
        ]

        with multiprocessing.Pool(n_process) as pool:
            shard_results = list(tqdm.tqdm(pool.imap(map_func, map_args), total=len(map_args)))

        res = shard_results[0]
        for i in range(1, len(shard_results)):
            res += shard_results[i]

        write_out_matrices(res, outdir)


def _main():
    # Pull out arguments
    args = parser.parse_args()
    matrix_generator = MatrixGenerator(
        a3m_dir=args.a3m_dir,
        transitions_dir=args.transitions_dir,
        n_process=args.n_process,
        expected_number_of_MSAs=args.expected_number_of_MSAs,
        outdir=args.outdir,
        max_families=args.max_families,
        num_sites=args.num_sites,
    )
    matrix_generator.run()


if __name__ == "__main__":
    _main()
