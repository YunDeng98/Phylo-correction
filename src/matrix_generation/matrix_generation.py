r"""
Reads transition edges and summarizes them into transition matrices based on branch
length quantization.
"""
import multiprocessing
import os
import sys

import logging
import numpy as np
import tqdm

import pandas as pd
import string
import random
import hashlib

from typing import List

sys.path.append("../")


def map_func(args: List) -> pd.DataFrame:
    # a3m_dir = args[0]
    transitions_dir = args[1]
    protein_family_names_for_shard = args[2]
    # outdir = args[3]
    alphabet = args[4]

    logger = logging.getLogger("matrix_generation")
    seed = int(
        hashlib.md5(("".join(protein_family_names_for_shard) + "matrix_generation").encode()).hexdigest()[:8], 16
    )
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
        logger.info("TODO: Filter transitions based on criteria! (low, short branches)")

        # Now add quantization column and group by it too.
        logger.info("TODO: Add quantization column and group by it too!")

        # Summarize remaining transitions into the matrices
        summarized_transitions = transitions_df.groupby(["starting_state", "ending_state"]).size()
        for starting_state in alphabet:
            for ending_state in alphabet:
                if (starting_state, ending_state) in summarized_transitions:
                    res.loc[starting_state, ending_state] += summarized_transitions[(starting_state, ending_state)]

    return res


def get_protein_family_names_for_shard(shard_id: int, n_process: int, protein_family_names: List[str]) -> List[str]:
    res = [protein_family_names[i] for i in range(len(protein_family_names)) if i % n_process == shard_id]
    return res


class MatrixGenerator:
    def __init__(
        self,
        a3m_dir: str,
        transitions_dir: str,
        n_process: int,
        expected_number_of_MSAs: int,
        outdir: str,
        max_families: int,
        num_sites: int,
        use_cached: bool = False,
    ):
        self.a3m_dir = a3m_dir
        self.transitions_dir = transitions_dir
        self.n_process = n_process
        self.expected_number_of_MSAs = expected_number_of_MSAs
        self.outdir = outdir
        self.max_families = max_families
        self.num_sites = num_sites
        self.use_cached = use_cached

    def run(self) -> None:
        a3m_dir = self.a3m_dir
        transitions_dir = self.transitions_dir
        n_process = self.n_process
        expected_number_of_MSAs = self.expected_number_of_MSAs
        outdir = self.outdir
        max_families = self.max_families
        num_sites = self.num_sites
        use_cached = self.use_cached

        logger = logging.getLogger("matrix_generation")

        # Caching pattern: skip any computation as soon as possible
        out_filepath = os.path.join(outdir, "matrices.txt")
        if use_cached and os.path.exists(out_filepath):
            logger.info(f"Skipping. Cached matrices for {transitions_dir} at {out_filepath}.")
            return

        assert num_sites in [1, 2]

        # Create list of amino acids
        amino_acids = [
            "A",
            "R",
            "N",
            "D",
            "C",
            "Q",
            "E",
            "G",
            "H",
            "I",
            "L",
            "K",
            "M",
            "F",
            "P",
            "S",
            "T",
            "W",
            "Y",
            "V",
        ] + ["-"]
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
            assert num_sites == 1
            alphabet = amino_acids[:]

        logger.info("Starting ... ")

        logger.info("TODO: Accept branch length quantization as input!")

        if os.path.exists(outdir) and not use_cached:
            raise ValueError(f"outdir {outdir} already exists. Aborting not to " f"overwrite!")
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if not os.path.exists(a3m_dir):
            raise ValueError(f"Could not find a3m_dir {a3m_dir}")

        filenames = sorted(list(os.listdir(a3m_dir)))
        if not len(filenames) == expected_number_of_MSAs:
            raise ValueError(
                f"Number of MSAs is {len(filenames)}, does not match " f"expected {expected_number_of_MSAs}"
            )
        protein_family_names = [x.split(".")[0] for x in filenames][:max_families]

        map_args = [
            [
                a3m_dir,
                transitions_dir,
                get_protein_family_names_for_shard(shard_id, n_process, protein_family_names),
                outdir,
                alphabet,
            ]
            for shard_id in range(n_process)
        ]

        if n_process > 1:
            with multiprocessing.Pool(n_process) as pool:
                shard_results = list(tqdm.tqdm(pool.imap(map_func, map_args), total=len(map_args)))
        else:
            shard_results = list(tqdm.tqdm(map(map_func, map_args), total=len(map_args)))

        res = shard_results[0]
        for i in range(1, len(shard_results)):
            res += shard_results[i]

        out_filepath = os.path.join(outdir, "matrices.txt")
        res.to_csv(out_filepath, sep="\t")

        os.system(f"chmod -R 555 {outdir}")
