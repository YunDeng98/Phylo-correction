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
    center = args[5]
    step_size = args[6]
    n_steps = args[7]
    keep_outliers = args[8]
    max_height = args[9]
    max_path_height = args[10]

    logger = logging.getLogger("phylo_correction.matrix_generation")
    seed = int(
        hashlib.md5(("".join(protein_family_names_for_shard) + "matrix_generation").encode()).hexdigest()[:8], 16
    )
    logger.info(f"Setting random seed to: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"Starting on {len(protein_family_names_for_shard)} families")

    # Compute grid of quantized branch lengths
    grid = np.array([center * (1.0 + step_size) ** i for i in range(-n_steps, n_steps + 1, 1)])
    # Create results data frame. There's one frequency matrix per quantized branch length.
    res = dict(
        [
            (grid_point_id,
             pd.DataFrame(
                 np.zeros(shape=(len(alphabet), len(alphabet)), dtype=int),
                 index=alphabet, columns=alphabet
             )
             )
            for grid_point_id in range(len(grid))
        ]
    )

    for protein_family_name in protein_family_names_for_shard:
        print(f"Starting on {protein_family_name}")
        transitions_df = pd.read_csv(os.path.join(transitions_dir, protein_family_name + ".transitions"), sep=",")

        # Filter transitions based on citeria
        transitions_df = transitions_df[transitions_df.height <= max_height]
        transitions_df = transitions_df[transitions_df.path_height <= max_path_height]

        # Assign edge lengths to closest quantized value (bucket). 'grid_point_id' is the index of the closest bucket,
        # which goes from 0 to len(grid) - 1 inclusive.
        tr_df_lengths = np.array(transitions_df.length)
        tr_df_grid_point_id = np.abs(tr_df_lengths[:, np.newaxis] - grid[np.newaxis, :]).argmin(axis=1)
        transitions_df['grid_point_id'] = tr_df_grid_point_id
        # Filter edges that are too short or too long if requested
        if not keep_outliers:
            # Determine if edge length is inside grid (i.e. if it is an outlier or not)
            tr_df_inside_grid = \
                ((tr_df_lengths[:, np.newaxis] - grid[np.newaxis, :]) <= 0).any(axis=1) \
                & ((tr_df_lengths[:, np.newaxis] - grid[np.newaxis, :]) >= 0).any(axis=1)
            transitions_df['inside_grid'] = tr_df_inside_grid
            # Filter!
            transitions_df = transitions_df[transitions_df.inside_grid]

        # Summarize remaining transitions into the frequency matrices
        summarized_transitions = transitions_df.groupby(["starting_state", "ending_state", "grid_point_id"]).size()
        for grid_point_id in range(len(grid)):
            for starting_state in alphabet:
                for ending_state in alphabet:
                    if (starting_state, ending_state, grid_point_id) in summarized_transitions:
                        res[grid_point_id].at[starting_state, ending_state] += summarized_transitions[(starting_state, ending_state, grid_point_id)]

    return res, grid


def get_protein_family_names_for_shard(shard_id: int, n_process: int, protein_family_names: List[str]) -> List[str]:
    res = [protein_family_names[i] for i in range(len(protein_family_names)) if i % n_process == shard_id]
    return res


class MatrixGenerator:
    r"""
    Generate frequency matrices from a transition file.

    The hyperparameters are passed in '__init__', and the frequency matrices are only
    computed upon call to the 'run' method.

    Args:
        a3m_dir: Directory where the MSA files (.a3m) are found. Although they
            are never read, this must be provided to be able to subsample families via the 'max_families' argument.
        transitions_dir: Directory where the transition files (.transitions) are found.
        n_process: Number of processes used to parallelize computation.
        expected_number_of_MSAs: The number of files in a3m_dir. This argument
            is only used to sanity check that the correct a3m_dir is being used.
            It has no functional implications.
        outdir: Directory where the generated frequency matrices will be found (matrices.txt file)
        max_families: Only run on 'max_families' randomly chosen files in a3m_dir.
            This is useful for testing and to see what happens if less data is used.
        num_sites: Whether the transitions are for single sites (num_sites=1) or for pairs
            of sites (num_sites=2).
        center: Quantization grid center
        step_size: Quantization grid step size (geometric)
        n_steps: Number of grid points left and right of center (for a total
            of 2 * n_steps + 1 grid points)
        keep_outliers: What to do with points that are outside the grid. If
            False, they will be dropped. If True, they will be assigned
            to the corresponding closest endpoint of the grid.
        max_height: Use only transitions whose starting node is at height
            at most max_height from the leaves in its subtree. This is
            used to filter out unreliable maximum parsimony transitions.
        max_path_height: Use only transitions whose starting node is at height
            at most max_path_height from the leaves in its subtree, in terms
            of the NUMBER OF EDGES. This is used to filter out unreliable
            maximum parsimony transitions.
        use_cached: If True and the output file already exists for a family,
            all computation will be skipped for that family.
    """
    def __init__(
        self,
        a3m_dir: str,
        transitions_dir: str,
        n_process: int,
        expected_number_of_MSAs: int,
        outdir: str,
        max_families: int,
        num_sites: int,
        center: float,
        step_size: float,
        n_steps: int,
        keep_outliers: bool,
        max_height: float,
        max_path_height: int,
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
        self.center = center
        self.step_size = step_size
        self.n_steps = n_steps
        self.keep_outliers = keep_outliers
        self.max_height = max_height
        self.max_path_height = max_path_height

    def run(self) -> None:
        a3m_dir = self.a3m_dir
        transitions_dir = self.transitions_dir
        n_process = self.n_process
        expected_number_of_MSAs = self.expected_number_of_MSAs
        outdir = self.outdir
        max_families = self.max_families
        num_sites = self.num_sites
        use_cached = self.use_cached
        center = self.center
        step_size = self.step_size
        n_steps = self.n_steps
        keep_outliers = self.keep_outliers
        max_height = self.max_height
        max_path_height = self.max_path_height

        logger = logging.getLogger("phylo_correction.matrix_generation")

        # Caching pattern: skip any computation as soon as possible
        out_filepath_total = os.path.join(outdir, "matrices.txt")
        out_filepath_quantized = os.path.join(outdir, "matrices_by_quantized_branch_length.txt")
        if use_cached and os.path.exists(out_filepath_total) and os.path.exists(out_filepath_quantized):
            logger.info(f"Skipping. Cached matrices for {transitions_dir} at {out_filepath_total} and {out_filepath_quantized}")
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
        ]

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

        if os.path.exists(outdir) and not use_cached:
            raise ValueError(f"outdir {outdir} already exists. Aborting not to " f"overwrite!")
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if not os.path.exists(a3m_dir):
            raise ValueError(f"Could not find a3m_dir {a3m_dir}")

        filenames = sorted(list(os.listdir(a3m_dir)))
        random.Random(123).shuffle(filenames)
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
                center,
                step_size,
                n_steps,
                keep_outliers,
                max_height,
                max_path_height,
            ]
            for shard_id in range(n_process)
        ]

        if n_process > 1:
            with multiprocessing.Pool(n_process) as pool:
                shard_results = list(tqdm.tqdm(pool.imap(map_func, map_args), total=len(map_args)))
        else:
            shard_results = list(tqdm.tqdm(map(map_func, map_args), total=len(map_args)))

        res, grid = shard_results[0]
        for i in range(1, len(shard_results)):
            res_i, _ = shard_results[i]
            for grid_point_id in range(len(grid)):
                res[grid_point_id] += res_i[grid_point_id]

        # "Symmetrize" the transition matrix if its on pair of sites. E.g. the transitions NG->NT and GN->TN should be unified
        if num_sites == 2:
            for grid_point_id in range(len(grid)):
                for aa1 in amino_acids:
                    for aa2 in amino_acids:
                        for aa3 in amino_acids:
                            for aa4 in amino_acids:
                                symmetrized_frequency = (res[grid_point_id].at[aa1 + aa2, aa3 + aa4] + res[grid_point_id].at[aa2 + aa1, aa4 + aa3]) / 2.0
                                res[grid_point_id].at[aa1 + aa2, aa3 + aa4] = symmetrized_frequency
                                res[grid_point_id].at[aa2 + aa1, aa4 + aa3] = symmetrized_frequency

        # Compute the total transitions, irrespective of branch length.
        res_all = res[0].copy()
        for grid_point_id in range(1, len(grid)):
            res_all += res[grid_point_id]
        res_all.to_csv(out_filepath_total, sep="\t")
        os.system(f"chmod 555 {out_filepath_total}")

        # Write out the frequency matrices for quantized branch lengths.
        for grid_point_id, grid_point in enumerate(grid):
            if grid_point_id == 0:
                with open(out_filepath_quantized, "w") as outfile:
                    outfile.write(f'{grid_point}\n')
            else:
                with open(out_filepath_quantized, "a") as outfile:
                    outfile.write(f'{grid_point}\n')
            res[grid_point_id].to_csv(out_filepath_quantized, mode='a', sep=' ', header=False, index=False)
        # Write out sentinel
        with open(out_filepath_quantized, "a") as outfile:
            outfile.write('9999.0\n')
        (0 * res[0]).to_csv(out_filepath_quantized, mode='a', sep=' ', header=False, index=False)
        os.system(f"chmod 555 {out_filepath_quantized}")
