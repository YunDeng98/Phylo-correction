import os
import logging
import numpy as np
import pandas as pd
import tempfile
from typing import Optional, Tuple

dir_path = os.path.dirname(os.path.realpath(__file__))
phyml_path = os.path.join(dir_path, 'phyml_github')
phyml_bin_path = os.path.join(dir_path, 'bin/phyml')


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
        logger.info(f"git clone https://github.com/stephaneguindon/phyml {phyml_path}")
        os.system(f"git clone https://github.com/stephaneguindon/phyml {phyml_path}")
        os.chdir(phyml_path)
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
    if rate_matrix_path is not None and model is not None:
        raise ValueError(f"Only one of rate_matrix_path and model can be provided. You provided:\n{rate_matrix_path}\nand\n{model}")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    phyml_log_filepath = os.path.join(outdir, "phyml_log.txt")
    os.system(f"pushd {outdir}")
    os.system(f"cp {input_msa_path} {outdir}")
    protein_family_name = input_msa_path.split('/')[-1]
    input_msa_path = os.path.join(outdir, protein_family_name)
    command = \
        f"{phyml_bin_path} " \
        f"--input {input_msa_path} " \
        f"--datatype aa " \
        f"--nclasses {num_rate_categories} " \
        f"--pinv e "
    if not optimize_rates:
        command += "-o tl "
    command += \
        f"--r_seed {random_seed} " \
        f"--bootstrap 0 " \
        f"-f m " \
        f"--alpha e " \
        f"--print_site_lnl "
    if model is not None:
        command += f"--model {model} "
    if rate_matrix_path is not None:
        command += \
            f"--model custom " \
            f"--aa_rate_file {rate_matrix_PAML} "
    command += f"> {phyml_log_filepath}"
    os.system(command)
    os.system("popd")
    phyml_stats_filepath = os.path.abspath(os.path.join(outdir, protein_family_name + "_phyml_stats.txt"))
    phyml_site_ll_filepath = os.path.abspath(os.path.join(outdir, protein_family_name + "_phyml_lk.txt"))
    if not os.path.exists(phyml_stats_filepath) or not os.path.exists(phyml_site_ll_filepath):
        raise Exception(f"PhyML failed to run. Files:\n{phyml_stats_filepath}\nAnd\n{phyml_site_ll_filepath}\ndo not both exist.\nCommand:\n{command}\n")
    return phyml_stats_filepath, phyml_site_ll_filepath


def get_phyml_ll(phyml_stats_filepath: str) -> float:
    for line in open(phyml_stats_filepath, "r"):
        if "Log-likelihood:" in line:
            return float(line.split()[-1])
    raise Exception(f"Could not parse Log-likelihood from file:\n{phyml_stats_filepath}")


def get_phyml_site_ll(phyml_site_ll_filepath: str) -> float:
    res = []
    started = False
    for line in open(phyml_site_ll_filepath, "r"):
        if started:
            res.append(float(line.split()[1]))
        if line.startswith("Site"):
            started = True
    return res


def get_number_of_taxa(phyml_stats_filepath: str) -> int:
    for line in open(phyml_stats_filepath, "r"):
        if "Number of taxa:" in line:
            return int(line.split()[-1])
    raise Exception(f"Could not parse Number of taxa from file:\n{phyml_stats_filepath}")


def get_number_of_sites(phyml_site_ll_filepath: str) -> int:
    return len(get_phyml_site_ll(phyml_site_ll_filepath))


def reproduce_Treebase_JTT_WAG_LG(
    treebase_dir: str,
    verbose: bool = False,
):
    filenames = sorted(list(os.listdir(treebase_dir)))
    rows = []
    for filename in filenames:
        protein_family_name = filename.split('.')[0]
        if verbose:
            print(f"Processing: {filename}")
        abspath = os.path.join(treebase_dir, filename)
        row = {}
        for model in ["JTT", "WAG", "LG"]:
            if verbose:
                print(f"Processing: {protein_family_name} with {model}")
            with tempfile.TemporaryDirectory() as phyml_outdir:
                phyml_stats_filepath, phyml_site_ll_filepath = run_phyml(
                    rate_matrix_path=None,
                    model=model,
                    input_msa_path=abspath,
                    num_rate_categories=4,
                    random_seed=0,
                    optimize_rates=True,
                    outdir=phyml_outdir,
                )
                row[model] = get_phyml_ll(phyml_stats_filepath)
                row["Name"] = protein_family_name
                row["Tax"] = get_number_of_taxa(phyml_stats_filepath)
                row["Sites"] = get_number_of_sites(phyml_site_ll_filepath)
        rows.append(row)
    res = pd.DataFrame(rows, columns=["Name", "Tax", "Sites", "JTT", "WAG", "LG"])
    res.sort_values(by=["Name"], inplace=True)
    res.reset_index(inplace=True)
    return res
