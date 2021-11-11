import logging
import multiprocessing
import os
import sys
import tempfile
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tqdm
import wget

from src.benchmarking import caching
from src.phyml.phyml import (
    get_number_of_sites,
    get_number_of_taxa,
    get_phyml_ll,
    run_phyml,
)
from src.utils import pushd, verify_integrity_of_directory


def init_logger():
    logger = logging.getLogger("phylo_correction.lg_paper")
    logger.setLevel(logging.DEBUG)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


init_logger()
logger = logging.getLogger("phylo_correction.lg_paper")


def wget_tarred_data_and_chmod(
    url: str,
    destination_directory: str,
    expected_number_of_files: int,
    mode: str = "555",
):
    """
    Gets tarred data from url into destination_directory and chmods the data to
    555 (or the mode specified) so that it is write protected.
    """
    destination_directory = os.path.abspath(destination_directory)
    if (
        os.path.exists(destination_directory)
        and len(os.listdir(destination_directory)) > 0
    ):
        verify_integrity_of_directory(
            dirpath=destination_directory,
            expected_number_of_files=expected_number_of_files,
            mode=mode,
        )
        logger.info(
            f"{url} has already been downloaded successfully to {destination_directory}. Not downloading again."
        )
        return
    logger.info(f"Downloading {url} into {destination_directory}")
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    logger.info(f"pushd into {destination_directory}")
    with pushd(destination_directory):
        wget.download(url)
        logger.info(f"wget {url} into {destination_directory}")
        os.system("tar -xvzf *.tar.gz >/dev/null")
        logger.info("Untarring file ...")
        os.system("rm *.tar.gz")
        logger.info("Removing tar file ...")
    os.system(f"chmod -R {mode} {destination_directory}")
    verify_integrity_of_directory(
        dirpath=destination_directory,
        expected_number_of_files=expected_number_of_files,
        mode=mode,
    )
    logger.info("Success!")


def get_lg_TreeBase_data(
    destination_directory: str,
):
    wget_tarred_data_and_chmod(
        url="http://www.atgc-montpellier.fr/download/datasets/models/lg_TreeBase.tar.gz",
        destination_directory=destination_directory,
        expected_number_of_files=59,
        mode="555",
    )


def get_lg_PfamTestingAlignments_data(
    destination_directory: str,
):
    wget_tarred_data_and_chmod(
        url="http://www.atgc-montpellier.fr/download/datasets/models/lg_PfamTestingAlignments.tar.gz",
        destination_directory=destination_directory,
        expected_number_of_files=500,
        mode="555",
    )


def _convert_lg_data(
    LG_training_data_dir: str,
    destination_directory: str,
):
    """
    Converts LG MSA data from the PHYLIP format to our format.
    """
    logger.info("Converting LG Training data to correct MSA format ...")
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    protein_family_names = sorted(list(os.listdir(LG_training_data_dir)))
    for protein_family_name in protein_family_names:
        with open(
            os.path.join(LG_training_data_dir, protein_family_name), "r"
        ) as file:
            res = ""
            lines = file.read().split("\n")
            n_seqs, n_sites = map(int, lines[0].split(" "))
            for i in range(n_seqs):
                line = lines[2 + i]
                try:
                    protein_name, protein_sequence = line.split()
                except:
                    raise ValueError(
                        f"For protein family {protein_family_name} , could not split line: {line}"
                    )
                assert len(protein_sequence) == n_sites
                res += f">{protein_name}\n"
                res += f"{protein_sequence}\n"
            output_filename = os.path.join(
                destination_directory,
                protein_family_name.replace(".", "_") + ".a3m",
            )
            with open(output_filename, "w") as outfile:
                outfile.write(res)
                outfile.flush()
            os.system(f"chmod 555 {output_filename}")


def get_lg_PfamTrainingAlignments_data(
    destination_directory: str,
):
    url = "http://www.atgc-montpellier.fr/download/datasets/models/lg_PfamTrainingAlignments.tar.gz"
    if (
        os.path.exists(destination_directory)
        and len(os.listdir(destination_directory)) > 0
    ):
        verify_integrity_of_directory(
            dirpath=destination_directory,
            expected_number_of_files=3912,
            mode="555",
        )
        logger.info(
            f"{url} has already been downloaded successfully to {destination_directory}. Not downloading again."
        )
        return
    with tempfile.TemporaryDirectory() as destination_directory_unprocessed:
        wget_tarred_data_and_chmod(
            url=url,
            destination_directory=destination_directory_unprocessed,
            expected_number_of_files=1,
            mode="777",
        )
        _convert_lg_data(
            LG_training_data_dir=os.path.join(
                destination_directory_unprocessed, "AllData"
            ),
            destination_directory=destination_directory,
        )
    verify_integrity_of_directory(
        dirpath=destination_directory,
        expected_number_of_files=3912,
        mode="555",
    )


@caching.cached()
def run_phyml_cached(
    input_msa_path: str,
    model: Optional[str],
    rate_matrix_path: Optional[str],
    num_rate_categories: int,
    random_seed: int,
    optimize_rates: bool,
) -> Tuple[str, str]:
    """
    Return the phyml_stats and phyml_site_ll obtained from running PhyML.
    """
    with tempfile.TemporaryDirectory() as phyml_outdir:
        phyml_stats_filepath, phyml_site_ll_filepath = run_phyml(
            rate_matrix_path=rate_matrix_path,
            model=model,
            input_msa_path=input_msa_path,
            num_rate_categories=num_rate_categories,
            random_seed=random_seed,
            optimize_rates=optimize_rates,
            outdir=phyml_outdir,
        )
        phyml_stats = open(phyml_stats_filepath).read()
        phyml_site_ll = open(phyml_site_ll_filepath).read()
        if len(phyml_stats) < 10 or len(phyml_site_ll) < 10:
            raise Exception(
                f"PhyML failed, returning:\n{phyml_stats}\n{phyml_site_ll}\n"
            )
        return phyml_stats, phyml_site_ll


def reproduce_Treebase_JTT_WAG_LG(
    treebase_dir: str,
    num_rate_categories: int = 4,
    random_seed: int = 0,
    optimize_rates: bool = True,
    max_families: int = 100000000,
    verbose: bool = False,
) -> pd.DataFrame:
    filenames = sorted(list(os.listdir(treebase_dir)))
    rows = []
    for filename in filenames[:max_families]:
        protein_family_name = filename.split(".")[0]
        input_msa_path = os.path.join(treebase_dir, filename)
        row = {}
        for model in ["JTT", "WAG", "LG"]:
            if verbose:
                print(f"Processing: {filename} with {model}")
            row[model_name] = get_phyml_ll(
                rate_matrix_path=rate_matrix_path,
                model=model,
                input_msa_path=input_msa_path,
                num_rate_categories=num_rate_categories,
                random_seed=random_seed,
                optimize_rates=optimize_rates,
            )
            row[model] = get_phyml_ll(phyml_stats)
            row["Name"] = protein_family_name
            row["Tax"] = get_number_of_taxa(phyml_stats)
            row["Sites"] = get_number_of_sites(phyml_site_ll)
        rows.append(row)
    res = pd.DataFrame(
        rows, columns=["Name", "Tax", "Sites", "JTT", "WAG", "LG"]
    )
    res.sort_values(by=["Name"], inplace=True)
    res.reset_index(inplace=True)
    return res


def _run_phyml_cached(args) -> Tuple[float, int, int]:
    """
    Multiprocessing wrapper to call run_phyml_cached
    """
    phyml_stats, phyml_site_ll = run_phyml_cached(
        input_msa_path=args[0],
        model=args[1],
        rate_matrix_path=args[2],
        num_rate_categories=args[3],
        random_seed=args[4],
        optimize_rates=args[5],
    )
    return (
        get_phyml_ll(phyml_stats),
        get_number_of_taxa(phyml_stats),
        get_number_of_sites(phyml_site_ll),
    )


def reproduce_Treebase_JTT_WAG_LG_parallel(
    treebase_dir: str,
    num_rate_categories: int = 4,
    random_seed: int = 0,
    optimize_rates: bool = True,
    max_families: int = 100000000,
    verbose: bool = False,
    n_process: int = 32,
) -> pd.DataFrame:
    """
    Wrapper on top of reproduce_Treebase_JTT_WAG_LG that
    uses multiprocessing the speed up computation.
    """
    # First warm up the cache in parallel
    map_args = []
    filenames = sorted(list(os.listdir(treebase_dir)))
    for filename in filenames[:max_families]:
        input_msa_path = os.path.join(treebase_dir, filename)
        for model in ["JTT", "WAG", "LG"]:
            rate_matrix_path = None
            map_args.append(
                (
                    input_msa_path,
                    model,
                    rate_matrix_path,
                    num_rate_categories,
                    random_seed,
                    optimize_rates,
                )
            )
    map_func = _run_phyml_cached
    if n_process > 1:
        with multiprocessing.Pool(n_process) as pool:
            list(tqdm.tqdm(pool.imap(map_func, map_args), total=len(map_args)))
    else:
        list(tqdm.tqdm(map(map_func, map_args), total=len(map_args)))

    # Now get what we want
    return reproduce_Treebase_JTT_WAG_LG(
        treebase_dir=treebase_dir,
        num_rate_categories=num_rate_categories,
        random_seed=random_seed,
        optimize_rates=optimize_rates,
        max_families=max_families,
        verbose=verbose,
    )
