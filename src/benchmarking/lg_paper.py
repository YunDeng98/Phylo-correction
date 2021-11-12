import logging
import multiprocessing
import os
import sys
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tqdm
import wget

from src.benchmarking import caching
from src.phyml.phyml import (
    get_phyml_ll_from_phyml_stats,
    run_phyml,
)
from src.utils import pushd, verify_integrity_of_directory


MODELS = [
    # ("JTT", "JTT", None),
    ("r__JTT", "r__JTT", None),
    # ("WAG", "WAG", None),
    ("r__WAG", "r__WAG", None),
    ("Cherry1nosr", None, "./input_data/Q1nosr.PAML.txt"),
    ("r__WAG'", "r__WAG'", None),
    ("Cherry_mixed", None, "./input_data/synthetic_rate_matrices/PAML/Q_learnt_from_LG_no_site_rates.PAML.txt"),
    ("r___WAG+LGF", "r__WAG+LG FRE", None),
    # ("WAG_PAML", None, "./input_data/synthetic_rate_matrices/PAML/WAG.PAML.txt"),
    ("r__LG", "r__LG", None),

    ("XRATE1", None, "./input_data/Q1XRATE.PAML.txt"),
    ("JTT-IPW1", None, "./input_data/Q1JTT_IPW.PAML.txt"),
    ("Parsimony1", None, "./input_data/Q1Parsimony.PAML.txt"),
    ("Cherry1_EQU", None, "./input_data/Q1EQU.PAML.txt"),
    ("Cherry1", None, "./input_data/Q1.PAML.txt"),
    ("Cherry1_div2", None, "./input_data/Q1div2.PAML.txt"),
    ("Cherry1_div4", None, "./input_data/Q1div4.PAML.txt"),
    ("Cherry1_div8", None, "./input_data/Q1div8.PAML.txt"),
    ("Cherry1_div16", None, "./input_data/Q1div16.PAML.txt"),
    ("Cherry1_div128", None, "./input_data/Q1div128.PAML.txt"),
    ("Cherry1_div1024", None, "./input_data/Q1div1024.PAML.txt"),

    ("JTT-IPW2", None, "./input_data/Q2JTT_IPW.PAML.txt"),
    ("Parsimony2", None, "./input_data/Q2Parsimony.PAML.txt"),
    ("Cherry2_EQU", None, "./input_data/Q2EQU.PAML.txt"),
    ("Cherry2", None, "./input_data/Q2.PAML.txt"),
    ("Cherry2_div2", None, "./input_data/Q2div2.PAML.txt"),
    ("Cherry2_div4", None, "./input_data/Q2div4.PAML.txt"),
    ("Cherry2_div8", None, "./input_data/Q2div8.PAML.txt"),
    ("Cherry2_div16", None, "./input_data/Q2div16.PAML.txt"),
    ("Cherry2_div128", None, "./input_data/Q2div128.PAML.txt"),
    # ("Cherry2_div1024", None, "./input_data/Q2div1024.PAML.txt"),  # ==> Poor estimate, leads to numeric errors in phyml

    ("LG", "LG", None),
    ("EQU", None, "./input_data/synthetic_rate_matrices/PAML/EQU.PAML.txt"),
    # ("Cherry2", None, "./input_data/Q2.PAML.txt"),
    # ("Cherry2v2", None, "./input_data/Q2v2.PAML.txt"),
    # ("Cherry3", None, "./input_data/Q3.PAML.txt"),
    # ("Cherry4", None, "./input_data/Q4.PAML.txt"),
    # ("Cherry", None, "./input_data/synthetic_rate_matrices/PAML/Q_learnt_from_LG.PAML.txt"),
    # ("Cherry_LG_nosr", None, "./input_data/synthetic_rate_matrices/PAML/Q_learnt_from_LG_no_site_rates.PAML.txt"),
    # ("LG_PAML", None, "./input_data/synthetic_rate_matrices/PAML/lg_LG.PAML.txt"),
    # ("Cherry_PFAM_32", None, "./input_data/synthetic_rate_matrices/PAML/Q_learnt_from_PFAM_32.PAML.txt"),
    # ("Cherry_PFAM_32_nosr", None, "./input_data/synthetic_rate_matrices/PAML/Q_learnt_from_PFAM_32_no_site_rates.PAML.txt"),
]


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


def get_phyml_ll(
    input_msa_path: str,
    model: Optional[str],
    rate_matrix_path: Optional[str],
    num_rate_categories: int,
    random_seed: int,
    optimize_rates: bool,
    pfam_or_treebase: Optional[str] = None,
) -> float:
    """
    This wrapper on top of run_phyml_cached that diverts logic to
    get the metrics reported by the LG paper.
    """
    if model is not None and model.startswith("r__"):
        if pfam_or_treebase is None:
            raise Exception("pfam_or_treebase should be provided to determine where to read the reported results from.")
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_name = model.split('__')[-1]
        if pfam_or_treebase == "treebase":
            df = pd.read_csv(os.path.join(dir_path, "Treebase.txt"), sep="\t")
        elif pfam_or_treebase == "pfam":
            df = pd.read_csv(os.path.join(dir_path, "Pfam.txt"), sep="\t")
        else:
            raise ValueError(f"pfam_or_treebase should be 'pfam' or 'treebase'. You provided: {pfam_or_treebase}")
        df = df.drop(0)
        df.set_index(["Name"], inplace=True)
        protein_family_name = input_msa_path.split('/')[-1].split('.')[0]
        return df.loc[protein_family_name, model_name]
    else:
        phyml_stats, _ = run_phyml_cached(
            rate_matrix_path=rate_matrix_path,
            model=model,
            input_msa_path=input_msa_path,
            num_rate_categories=num_rate_categories,
            random_seed=random_seed,
            optimize_rates=optimize_rates,
        )
        return get_phyml_ll_from_phyml_stats(phyml_stats)


def get_filenames_Treebase(treebase_dir: str) -> List[str]:
    # We sort families by number of taxa to ease testing with max_families
    filenames = ['M2344', 'M1882', 'M1379', 'M1381', 'M1380', 'M1382', 'M964', 'M1046', 'M1496', 'M1384', 'M1385', 'M1383', 'M1497', 'M1502', 'M1498', 'M1506', 'M1374', 'M1507', 'M1989', 'M658', 'M2640', 'M2641', 'M2638', 'M1378', 'M2639', 'M2637', 'M1377', 'M1023', 'M2636', 'M1508', 'M1499', 'M1373', 'M1372', 'M1503', 'M2302', 'M1500', 'M2558', 'M1993', 'M1335', 'M2478', 'M2479', 'M2476', 'M1768', 'M2304', 'M1812', 'M1990', 'M1291', 'M1601', 'M1376', 'M1504', 'M2477', 'M2480', 'M2577', 'M337', 'M1487', 'M931', 'M730', 'M1392', 'M686']
    # Put the two families with massive number of sites at the end.
    filenames = [x for x in filenames if x not in ["M2577", "M964"]] + ["M2577", "M964"]
    filenames = [x + '.clean.ProtGB_phyml' for x in filenames]
    # filenames = sorted(list(os.listdir(treebase_dir)))
    return filenames


def get_filenames_Pfam(pfam_dir: str) -> List[str]:
    filenames = sorted(list(os.listdir(pfam_dir)))
    return filenames


def reproduce_JTT_WAG_LG_table(
    a3m_phylip_dir: str,
    filenames: List[str],
    num_rate_categories: int = 4,
    random_seed: int = 0,
    optimize_rates: bool = True,
    max_families: int = 100000000,
    verbose: bool = False,
    pfam_or_treebase: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    for filename in filenames[:max_families]:
        protein_family_name = filename.split(".")[0]
        input_msa_path = os.path.join(a3m_phylip_dir, filename)
        row = {}
        for (model_name, model, rate_matrix_path) in MODELS:
            if verbose:
                print(f"Processing: {filename} with {model}")
            row[model_name] = get_phyml_ll(
                rate_matrix_path=rate_matrix_path,
                model=model,
                input_msa_path=input_msa_path,
                num_rate_categories=num_rate_categories,
                random_seed=random_seed,
                optimize_rates=optimize_rates,
                pfam_or_treebase=pfam_or_treebase,
            )
            row["Name"] = protein_family_name
        rows.append(row)
    res = pd.DataFrame(
        rows,
        columns=[
            "Name",
        ] + [model[0] for model in MODELS]
    )
    res.sort_values(by=["Name"], inplace=True)
    res.reset_index(inplace=True)
    return res


def _run_phyml_cached(args) -> Tuple[float, int, int]:
    """
    Multiprocessing wrapper to call run_phyml_cached
    """
    input_msa_path = args[0]
    model = args[1]
    rate_matrix_path = args[2]
    num_rate_categories = args[3]
    random_seed = args[4]
    optimize_rates = args[5]
    pfam_or_treebase = args[6]
    return get_phyml_ll(
        rate_matrix_path=rate_matrix_path,
        model=model,
        input_msa_path=input_msa_path,
        num_rate_categories=num_rate_categories,
        random_seed=random_seed,
        optimize_rates=optimize_rates,
        pfam_or_treebase=pfam_or_treebase,
    )


def reproduce_JTT_WAG_LG_table_parallel(
    a3m_phylip_dir: str,
    filenames: List[str],
    num_rate_categories: int = 4,
    random_seed: int = 0,
    optimize_rates: bool = True,
    max_families: int = 100000000,
    verbose: bool = False,
    n_process: int = 32,
    pfam_or_treebase: Optional[str] = None,
) -> pd.DataFrame:
    """
    Wrapper on top of reproduce_JTT_WAG_LG_table that
    uses multiprocessing the speed up computation.
    """
    # First warm up the cache in parallel
    map_args = []
    for filename in filenames[:max_families]:
        input_msa_path = os.path.join(a3m_phylip_dir, filename)
        for (_, model, rate_matrix_path) in MODELS:
            map_args.append(
                (
                    input_msa_path,
                    model,
                    rate_matrix_path,
                    num_rate_categories,
                    random_seed,
                    optimize_rates,
                    pfam_or_treebase,
                )
            )
    map_func = _run_phyml_cached
    if n_process > 1:
        with multiprocessing.Pool(n_process) as pool:
            list(tqdm.tqdm(pool.imap(map_func, map_args), total=len(map_args)))
    else:
        list(tqdm.tqdm(map(map_func, map_args), total=len(map_args)))

    # Now get what we want
    return reproduce_JTT_WAG_LG_table(
        a3m_phylip_dir=a3m_phylip_dir,
        filenames=filenames,
        num_rate_categories=num_rate_categories,
        random_seed=random_seed,
        optimize_rates=optimize_rates,
        max_families=max_families,
        verbose=verbose,
        pfam_or_treebase=pfam_or_treebase,
    )


def reproduce_JTT_WAG_LG_table_parallel_Treebase(
    treebase_dir: str,
    num_rate_categories: int = 4,
    random_seed: int = 0,
    optimize_rates: bool = True,
    max_families: int = 100000000,
    verbose: bool = False,
    n_process: int = 32,
) -> pd.DataFrame:
    return reproduce_JTT_WAG_LG_table_parallel(
        a3m_phylip_dir=treebase_dir,
        filenames=get_filenames_Treebase(treebase_dir=treebase_dir),
        num_rate_categories=num_rate_categories,
        random_seed=random_seed,
        optimize_rates=optimize_rates,
        max_families=max_families,
        verbose=verbose,
        n_process=n_process,
        pfam_or_treebase="treebase",
    )


def reproduce_JTT_WAG_LG_table_parallel_Pfam(
    pfam_dir: str,
    num_rate_categories: int = 4,
    random_seed: int = 0,
    optimize_rates: bool = True,
    max_families: int = 100000000,
    verbose: bool = False,
    n_process: int = 32,
) -> pd.DataFrame:
    return reproduce_JTT_WAG_LG_table_parallel(
        a3m_phylip_dir=pfam_dir,
        filenames=get_filenames_Pfam(pfam_dir=pfam_dir),
        num_rate_categories=num_rate_categories,
        random_seed=random_seed,
        optimize_rates=optimize_rates,
        max_families=max_families,
        verbose=verbose,
        n_process=n_process,
        pfam_or_treebase="pfam",
    )
