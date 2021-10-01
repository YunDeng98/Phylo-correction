r"""
Run XRATE on stockholm files.
"""
import multiprocessing
import os

import logging
import tqdm
import numpy as np
import pandas as pd
import sys

from typing import List


from src.utils import subsample_protein_families
sys.path.append("../")
import Phylo_util


def normalized(Q):
    pi = Phylo_util.solve_stationery_dist(Q)
    mutation_rate = pi @ -np.diag(Q)
    return Q / mutation_rate


def install_xrate():
    """
    See http://biowiki.org/wiki/index.php/Xrate_Software.
    """
    logger = logging.getLogger("phylo_correction.xrate")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    xrate_path = os.path.join(dir_path, 'x_rate_github')
    xrate_bin_path = os.path.join(xrate_path, 'bin/xrate')
    logger.info("Checking for XRATE ...")
    if not os.path.exists(xrate_bin_path):
        # TODO: Make this part of installation?
        logger.info(f"git clone https://github.com/ihh/dart {xrate_path}")
        os.system(
            f"git clone https://github.com/ihh/dart {xrate_path}"
        )
        logger.info(f"cd {xrate_path} ...")
        os.chdir(xrate_path)
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info("./configure --without-guile ...")
        os.system("./configure --without-guile")
        logger.info("make xrate ...")
        os.system("make xrate")
        logger.info("Done!")
    if not os.path.exists(xrate_bin_path):
        raise ValueError(f"Failed to install XRATE")


def run_xrate(
    stock_input_paths: List[str],
    output_path: str,
    estimate_trees: bool = False,
):
    logger = logging.getLogger("phylo_correction.xrate")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    xrate_path = os.path.join(dir_path, 'x_rate_github')
    xrate_bin_path = os.path.join(xrate_path, 'bin/xrate')
    if estimate_trees:
        cmd = f"{xrate_bin_path} {' '.join(stock_input_paths)} -e {xrate_path}/grammars/nullprot.eg -g {xrate_path}/grammars/nullprot.eg -t {output_path} -log 6"
    else:
        cmd = f"{xrate_bin_path} {' '.join(stock_input_paths)} -g {xrate_path}/grammars/nullprot.eg -t {output_path} -log 6"
    logger.info(f"Running {cmd}")
    os.system(cmd)


def xrate_to_numpy(xrate_output_file: str) -> np.array:
    amino_acids = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    res_df = pd.DataFrame(np.zeros(shape=(len(amino_acids), len(amino_acids))), index=amino_acids, columns=amino_acids)
    with open(xrate_output_file, "r") as file:
        lines = list(file)
        for line in lines:
            if line.startswith("  (mutate (from (") and "rate" in line:
                aa1 = line[17].upper()
                aa2 = line[26].upper()
                rate = float(line.replace(')', '').split(' ')[-1])
                res_df.loc[aa1, aa2] = rate
                res_df.loc[aa1, aa1] -= rate
    return res_df.to_numpy()


class XRATE:
    r"""
    Generate input for XRATE, given the MSAs and trees.

    The hyperparameters are passed in '__init__', and the outputs are only
    computed upon call to the 'run' method.

    Args:
        a3m_dir_full: Directory with MSAs for ALL protein families. Used
            to determine which max_families will get subsampled.
        xrate_input_dir: Directory where the stockholm files (.stock) are found.
        expected_number_of_MSAs: The number of files in a3m_dir. This argument
            is only used to sanity check that the correct a3m_dir is being used.
            It has no functional implications.
        outdir: Directory where the learned rate matrices will be found.
        max_families: Only run on 'max_families' randomly chosen files in a3m_dir_full.
            This is useful for testing and to see what happens if less data is used.
        use_cached: If True and the output file already exists for a family,
            all computation will be skipped for that family.
    """
    def __init__(
        self,
        a3m_dir_full: str,
        xrate_input_dir: str,
        expected_number_of_MSAs: int,
        outdir: str,
        max_families: int,
        use_cached: bool = False,
    ):
        self.a3m_dir_full = a3m_dir_full
        self.xrate_input_dir = xrate_input_dir
        self.expected_number_of_MSAs = expected_number_of_MSAs
        self.outdir = outdir
        self.max_families = max_families
        self.use_cached = use_cached

    def run(self) -> None:
        logger = logging.getLogger("phylo_correction.xrate")
        logger.info(f"Starting on max_families={self.max_families}, outdir: {self.outdir}")

        a3m_dir_full = self.a3m_dir_full
        xrate_input_dir = self.xrate_input_dir
        expected_number_of_MSAs = self.expected_number_of_MSAs
        outdir = self.outdir
        max_families = self.max_families
        use_cached = self.use_cached

        learned_matrix_path = os.path.join(self.outdir, "learned_matrix.txt")
        normalized_learned_matrix_path = os.path.join(self.outdir, "learned_matrix_normalized.txt")
        if os.path.exists(learned_matrix_path) and os.path.exists(normalized_learned_matrix_path) and use_cached:
            # logger.info(f"Skipping. Cached XRATE results at {outdir}")
            return

        # Create output directory
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if not os.path.exists(a3m_dir_full):
            raise ValueError(f"Could not find a3m_dir_full {a3m_dir_full}")

        protein_family_names = subsample_protein_families(
            a3m_dir_full,
            expected_number_of_MSAs,
            max_families
        )

        run_xrate(
            stock_input_paths=[
                os.path.join(xrate_input_dir, f"{protein_family_name}.stock") for protein_family_name in protein_family_names
            ],
            output_path=os.path.join(outdir, "learned_matrix.xrate")
        )
        Q = xrate_to_numpy(xrate_output_file=os.path.join(outdir, "learned_matrix.xrate"))
        np.savetxt(learned_matrix_path, Q)
        os.system(f"chmod 555 {learned_matrix_path}")

        np.savetxt(normalized_learned_matrix_path, normalized(Q))
        os.system(f"chmod 555 {normalized_learned_matrix_path}")
