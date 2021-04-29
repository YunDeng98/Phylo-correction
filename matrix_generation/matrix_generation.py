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
from typing import Dict, Tuple

from ete3 import Tree
import numpy as np
import pandas as pd
import string

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

amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'] + ['-']
for letter in string.ascii_uppercase:
    if letter not in amino_acids:
        amino_acids += [letter]
# amino_acids = list(string.ascii_uppercase) + ['-']
res = pd.DataFrame(np.zeros(shape=(len(amino_acids), len(amino_acids)), dtype=int), index=amino_acids, columns=amino_acids)

def map_func(args):
    logger = logging.getLogger("matrix_generation")
    a3m_dir = args[0]
    transitions_dir = args[1]
    protein_family_name = args[2]
    outdir = args[3]
    logger.info(f"Starting on family {protein_family_name}")
    print(f"COMPUTING MATRIX FOR {protein_family_name}")
    transitions_df = pd.read_csv(os.path.join(transitions_dir, protein_family_name + ".transitions"), sep=",")

    # Filter transitions based on citeria
    print(f"TODO: Filter transitions based on criteria! (low, short branches)")

    # Summarize remaining transitions into the matrices
    summarized_transitions = transitions_df.groupby(["starting_state", "ending_state"]).size()
    for starting_state in amino_acids:
        for ending_state in amino_acids:
            if (starting_state, ending_state) in summarized_transitions:
                res.loc[starting_state, ending_state] += summarized_transitions[(starting_state, ending_state)]


def write_out_matrices(outdir):
    out_filepath = os.path.join(outdir, "matrices.txt")
    res.to_csv(out_filepath, sep="\t")


if __name__ == "__main__":
    np.random.seed(1)

    # Pull out arguments
    args = parser.parse_args()
    a3m_dir = args.a3m_dir
    transitions_dir = args.transitions_dir
    n_process = args.n_process
    expected_number_of_MSAs = args.expected_number_of_MSAs
    outdir = args.outdir
    max_families = args.max_families

    init_logger()
    logger = logging.getLogger("matrix_generation")
    logger.info("Starting ... ")

    print(f"TODO: Accept branch length quantization as input!")
    print(f"TODO: Accept alphabet of states as input? Or: just infer from data whether we are single or double site.")

    if os.path.exists(outdir):
        raise ValueError(
            f"outdir {outdir} already exists. Aborting not to " f"overwrite!"
        )
    os.mkdir(outdir)

    if not os.path.exists(a3m_dir):
        raise ValueError(f"Could not find a3m_dir {a3m_dir}")

    filenames = list(os.listdir(a3m_dir))
    if not len(filenames) == expected_number_of_MSAs:
        raise ValueError(
            f"Number of MSAs is {len(filenames)}, does not match "
            f"expected {expected_number_of_MSAs}"
        )
    protein_family_names = [x.split(".")[0] for x in filenames][:max_families]

    # print(f"protein_family_names = {protein_family_names}")

    map_args = [
        (a3m_dir, transitions_dir, protein_family_name, outdir)
        for protein_family_name in protein_family_names
    ]
    # with multiprocessing.Pool(n_process) as pool:
    #     list(tqdm.tqdm(pool.imap(map_func, map_args), total=len(map_args)))

    for args in map_args:
        map_func(args)

    write_out_matrices(outdir)