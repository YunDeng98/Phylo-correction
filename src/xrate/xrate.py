r"""
http://biowiki.org/wiki/index.php/Xrate_Software

Run XRATE on stockholm files.
"""
import multiprocessing
import os
from os.path import dirname
import subprocess
import tempfile

import logging
import tqdm
import numpy as np
import pandas as pd
import sys

from typing import List, Optional


from src.xrate.xrate_input_generation import get_stock_filenames
from src.utils import subsample_protein_families, verify_integrity, pushd
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
        with pushd(xrate_path):
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
    xrate_grammar: Optional[str],
    output_path: str,
    logfile: Optional[str] = None,
    estimate_trees: bool = False,
):
    logger = logging.getLogger("phylo_correction.xrate")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    xrate_path = os.path.join(dir_path, 'x_rate_github')
    xrate_bin_path = os.path.join(xrate_path, 'bin/xrate')

    if xrate_grammar is None:
        xrate_grammar = f"{xrate_path}/grammars/nullprot.eg"

    if not os.path.exists(xrate_grammar):
        raise ValueError(f"Grammar file {xrate_grammar} does not exist.")

    if estimate_trees:
        cmd = f"{xrate_bin_path} {' '.join(stock_input_paths)} -e {xrate_grammar} -g {xrate_grammar} -log 6 -f 3 -t {output_path}"
    else:
        cmd = f"{xrate_bin_path} {' '.join(stock_input_paths)} -g {xrate_grammar} -log 6 -f 3 -t {output_path}"
    if logfile is not None:
        cmd += f" 2>&1 | tee {logfile}"
    # Write the command to a file and run it from there with bash because
    # running directly subprocess.run(cmd) fails due to command length limit.
    with tempfile.NamedTemporaryFile("w") as bash_script_file:
        bash_script_filename = bash_script_file.name
        bash_script_file.write(cmd)
        bash_script_file.flush()  # This is key, or else the call below will fail!
        logger.info(f"Running {cmd}")
        subprocess.run(f"bash {bash_script_filename}", shell=True, check=True)


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


def get_stock_filenames_for_training(
    stock_dir: str,
    protein_family_names: List[str],
    use_site_specific_rates: bool,
    num_rate_categories: int,
) -> List[str]:
    """
    Get filenames for training XRATE.

    The caveat is that when use_site_specific_rates=True, some of the files
    might contain empty MSAs because there were no sites with those rates, so we
    want to exclude them to reduce the command length, and avoid possible
    XRATE complaints.
    """
    filenames = get_stock_filenames(
        stock_dir=stock_dir,
        protein_family_names=protein_family_names,
        use_site_specific_rates=use_site_specific_rates,
        num_rate_categories=num_rate_categories,
    )
    training_filenames = []
    for filename in filenames:
        third_line = open(filename).read().split('\n')[2]
        msa_length = len(third_line.split(' ')[1])
        if msa_length == 0:
            # This file must be skept.
            if not use_site_specific_rates:
                # This should never happen!
                raise Exception(
                    "stock file has an empty MSA, and you are not using site"
                    "specific rates. This can never happen!"
                )
        else:
            training_filenames.append(filename)
    return training_filenames


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
        xrate_grammar: The XRATE grammar file containing the rate matrix
            parameterization and initialization.
        use_site_specific_rates: Whether to use site specific rates. When True,
            we get the LG method; when False, we get the WAG method.
        num_rate_categories: The number of rate categories, in case they shall
            be used (when use_site_specific_rates=True).
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
        xrate_grammar: Optional[str],
        use_site_specific_rates: bool,
        num_rate_categories: int,
        use_cached: bool = False,
    ):
        self.a3m_dir_full = a3m_dir_full
        self.xrate_input_dir = xrate_input_dir
        self.expected_number_of_MSAs = expected_number_of_MSAs
        self.outdir = outdir
        self.max_families = max_families
        self.xrate_grammar = xrate_grammar
        self.use_site_specific_rates = use_site_specific_rates
        self.num_rate_categories = num_rate_categories
        self.use_cached = use_cached

    def run(self) -> None:
        logger = logging.getLogger("phylo_correction.xrate")
        logger.info(f"Starting on max_families={self.max_families}, outdir: {self.outdir}")

        a3m_dir_full = self.a3m_dir_full
        xrate_input_dir = self.xrate_input_dir
        expected_number_of_MSAs = self.expected_number_of_MSAs
        outdir = self.outdir
        max_families = self.max_families
        xrate_grammar = self.xrate_grammar
        use_site_specific_rates = self.use_site_specific_rates
        num_rate_categories = self.num_rate_categories
        use_cached = self.use_cached

        # Caching pattern
        learned_matrix_path = os.path.join(self.outdir, "learned_matrix.txt")
        normalized_learned_matrix_path = os.path.join(self.outdir, "learned_matrix_normalized.txt")
        if os.path.exists(learned_matrix_path) and os.path.exists(normalized_learned_matrix_path) and use_cached:
            verify_integrity(learned_matrix_path)
            verify_integrity(normalized_learned_matrix_path)
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
            stock_input_paths=get_stock_filenames_for_training(
                stock_dir=xrate_input_dir,
                protein_family_names=protein_family_names,
                use_site_specific_rates=use_site_specific_rates,
                num_rate_categories=num_rate_categories,
            ),
            xrate_grammar=xrate_grammar,
            output_path=os.path.join(outdir, "learned_matrix.xrate"),
            logfile=os.path.join(outdir, "xrate_log"),
        )
        Q = xrate_to_numpy(xrate_output_file=os.path.join(outdir, "learned_matrix.xrate"))
        np.savetxt(learned_matrix_path, Q)
        os.system(f"chmod 555 {learned_matrix_path}")

        np.savetxt(normalized_learned_matrix_path, normalized(Q))
        os.system(f"chmod 555 {normalized_learned_matrix_path}")
