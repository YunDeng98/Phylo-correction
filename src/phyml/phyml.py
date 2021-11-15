import logging
import os
import sys
from typing import Optional, Tuple

import numpy as np
from src.utils import pushd

sys.path.append("../")
from Phylo_util import solve_stationery_dist

dir_path = os.path.dirname(os.path.realpath(__file__))
phyml_path = os.path.join(dir_path, "phyml_github")
phyml_bin_path = os.path.join(dir_path, "bin/phyml")


def install_phyml():
    """
    Install PhyML

    See https://github.com/stephaneguindon/phyml

    TODO: Enable MPI to accelerate? (Probably not worth it since we will multiprocess anyway)
    """
    logger = logging.getLogger("phylo_correction.phyml")
    logger.info("Checking for PhyML ...")
    if not os.path.exists(phyml_bin_path):
        # TODO: Make this part of installation?
        logger.info(
            f"git clone https://github.com/stephaneguindon/phyml {phyml_path}"
        )
        os.system(
            f"git clone https://github.com/stephaneguindon/phyml {phyml_path}"
        )
        with pushd(phyml_path):
            commands = [
                "bash ./autogen.sh",
                f"./configure --enable-phyml --prefix={dir_path}",
                "make",
                "make install",
            ]
            for command in commands:
                logger.info(command)
                os.system(command)
            logger.info("Done!")
    if not os.path.exists(phyml_bin_path):
        raise Exception("Failed to install PhyML")


def to_paml_format(
    input_rate_matrix_path: str,
    output_rate_matrix_path: str,
):
    """
    Convert a rate matrix into the PAML format required to run PhyML.
    """
    Q = np.loadtxt(input_rate_matrix_path)
    pi = solve_stationery_dist(Q)
    E, F = Q / pi, np.diag(pi)
    res = ""
    n = Q.shape[0]
    for i in range(n):
        for j in range(i):
            res += "%.6f " % E[i, j]
        res += "\n"
    res += "\n"
    for i in range(n):
        res += "%.6f " % F[i, i]
    with open(output_rate_matrix_path, "w") as outfile:
        outfile.write(res)
        outfile.flush()


def run_phyml(
    rate_matrix_path: Optional[str],
    model: Optional[str],
    input_msa_path: str,
    num_rate_categories: int,
    random_seed: int,
    optimize_rates: bool,
    outdir: str,
) -> Tuple[str, str]:
    """
    Run PhyML.

    Returns the phyml_stats_filepath and phyml_site_ll_filepath.
    Raises an Exception if PhyML fails to run.

    TODO: running with --freerates could be interesting!
    TODO: What is the sequential format?

    Args:
        rate_matrix_path: Path to the rate matrix in TODO format.
        model: E.g. LG, WAG, JTT, etc.
        input_msa_path: MSA in PHYLIP format.
        num_rate_categories: Number of discrete Gamma rate categories.
        random_seed: Random seed for PhyML determinism.
        outdir: Where to write the output.
    """
    if rate_matrix_path:
        rate_matrix_path = os.path.abspath(rate_matrix_path)
    input_msa_path = os.path.abspath(input_msa_path)
    outdir = os.path.abspath(outdir)
    if rate_matrix_path is not None and model is not None:
        raise ValueError(
            f"Only one of rate_matrix_path and model can be provided. You provided:\n{rate_matrix_path}\nand\n{model}"
        )
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    phyml_log_filepath = os.path.join(outdir, "phyml_log.txt")
    with pushd(outdir):
        input_msa_path_orig = input_msa_path
        os.system(f"cp {input_msa_path} {outdir}")
        protein_family_name = input_msa_path.split("/")[-1]
        input_msa_path = os.path.join(outdir, protein_family_name)
        command = (
            f"{phyml_bin_path} "
            f"--input {input_msa_path} "
            f"--datatype aa "
            f"--nclasses {num_rate_categories} "
            f"--pinv e "
        )
        if not optimize_rates:
            command += "-o tl "
        command += (
            f"--r_seed {random_seed} "
            f"--bootstrap 0 "
            f"-f m "
            f"--alpha e "
            f"--print_site_lnl "
        )
        if model is not None:
            command += f"--model {model} "
        if rate_matrix_path is not None:
            command += f"--model custom " f"--aa_rate_file {rate_matrix_path} "
        command += f"> {phyml_log_filepath}"
        os.system(command)
    phyml_stats_filepath = os.path.abspath(
        os.path.join(outdir, protein_family_name + "_phyml_stats.txt")
    )
    phyml_site_ll_filepath = os.path.abspath(
        os.path.join(outdir, protein_family_name + "_phyml_lk.txt")
    )
    if not os.path.exists(phyml_stats_filepath) or not os.path.exists(
        phyml_site_ll_filepath
    ):
        raise Exception(
            f"PhyML failed to run. Files:\n{phyml_stats_filepath}\nAnd\n{phyml_site_ll_filepath}\ndo not both exist.\nCommand:\n{command}\n"
        )
    phyml_stats = open(phyml_stats_filepath).read()
    phyml_site_ll = open(phyml_site_ll_filepath).read()
    if len(phyml_stats) < 10 or len(phyml_site_ll) < 10:
        raise Exception(
            f"PhyML failed, returning:\n{phyml_stats}\n{phyml_site_ll}\nCommand was:\n{command}\nHere\n{input_msa_path}\ncan be replaced by:\n{input_msa_path_orig}\n"
        )
    return phyml_stats_filepath, phyml_site_ll_filepath


# def get_phyml_ll_from_phyml_stats(phyml_stats: str) -> float:
#     """
#     Given the phyml_stats file contents, return the Log-likelihood.
#     """
#     for line in phyml_stats.split("\n"):
#         if "Log-likelihood:" in line:
#             return float(line.split()[-1])
#     # PhyML might have crashed due to numerical instability issue.
#     # Try to recover by getting the the last instance of LnL
#     # We reverse the lines in the loop below to get the last iteration first.
#     for line in phyml_stats.split("\n")[::-1]:
#         line_contents = line.split()
#         for i, x in enumerate(line_contents):
#             if x == "lnL=":
#                 logger = logging.getLogger("phylo_correction.phyml")
#                 logger.warning(
#                     "PhyML probably crashed due to numerical instability, but "
#                     "I was able to get the lnL of the last iteration."
#                 )
#                 return float(line_contents[i + 1])
#     raise Exception(
#         f"Could not parse Log-likelihood not lnL from file contents:\n{phyml_stats}"
#     )


def get_phyml_ll_from_phyml_stats(phyml_stats: str) -> float:
    """
    Given the phyml_stats file contents, return the Log-likelihood.
    """
    for line in phyml_stats.split("\n"):
        if "Log-likelihood:" in line:
            return float(line.split()[-1])
    raise Exception(
        f"Could not parse Log-likelihood from file contents:\n{phyml_stats}"
    )


def get_phyml_site_ll_from_phyml_site_ll(phyml_site_ll: str) -> float:
    """
    Given the phyml_site_ll file contents, return the Site Log-likelihood.
    """
    res = []
    started = False
    for line in phyml_site_ll.split("\n")[:-1]:
        if started:
            res.append(float(line.split()[1]))
        if line.startswith("Site"):
            started = True
    return res


def get_number_of_taxa_from_phyml_stats(phyml_stats: str) -> int:
    """
    Given the phyml_stats file contents, return the Log-likelihood.
    """
    for line in phyml_stats.split("\n"):
        if "Number of taxa:" in line:
            return int(line.split()[-1])
    raise Exception(
        f"Could not parse Number of taxa from file contents:\n{phyml_stats}"
    )


def get_number_of_sites_from_phyml_site_ll(phyml_site_ll: str) -> int:
    """
    Given the phyml_site_ll file contents, return the Site Log-likelihood.
    """
    return len(get_phyml_site_ll(phyml_site_ll))
