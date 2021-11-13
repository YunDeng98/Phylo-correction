import logging
import multiprocessing
import os
import sys
import tempfile
from typing import List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import wget

from src.benchmarking import caching
from src.phyml.phyml import get_phyml_ll_from_phyml_stats, run_phyml
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


def get_registered_models() -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    Get the list of registered models.

    Each model is characterized by a triple containing:
        1. The model name chosen by us (e.g. this is what will appear in the
            plots).
        2. The model name *in PhyML*, if it should be passed in as a parameter
            to PhyML, e.g. 'LG', 'WAG', 'JTT'. If the results REPORTED in the
            paper are desired instead, the model name should be prefixed with
            'r__', i.e. 'r__LG', 'r__WAG', 'r__LG'.
        3. The model rate matrix in PAML format, if it should be passed in as a
            parameter to PhyML
    Note that only one of (2) and (3) is thus provided. The other one should be
    None.
    """
    return [
        # # Reported & reproduced results
        ("JTT (reported)", "r__JTT", None),
        ("JTT (reproduced)", "JTT", None),
        ("WAG (reported)", "r__WAG", None),
        ("WAG (reproduced)", "WAG", None),
        # ("WAG+LGF (reported)", "r__WAG+LG FRE", None),
        ("WAG' (reported)", "r__WAG'", None),
        ("WAG' reproduced (w/XRATE)", None, "./input_data/Q1XRATE.PAML.txt"),
        ("LG (reported)", "r__LG", None),
        ("LG (reproduced)", "LG", None),
        # Our method, no site rates
        ("Cherry; No site rates", None, "./input_data/Q1nosr.PAML.txt"),
        # Our method
        ("Cherry", None, "./input_data/Q1.PAML.txt"),
        ("Cherry; 2nd iteration", None, "./input_data/Q2v2.PAML.txt"),
        # FastTree initialization
        ("Cherry; FastTree w/EQU", None, "./input_data/Q1EQU.PAML.txt"),
        (
            "Cherry; FastTree w/EQU; 2nd iteration",
            None,
            "./input_data/Q2EQU.PAML.txt",
        ),
        # Maximum Parsimony baseline (all edges)
        ("MP (all edges)", None, "./input_data/Q1Parsimony.PAML.txt"),
        (
            "MP (all edges); 2nd iteration",
            None,
            "./input_data/Q2Parsimony.PAML.txt",
        ),
        # JTT-IPW baseline
        ("JTT-IPW", None, "./input_data/Q1JTT_IPW.PAML.txt"),
        ("JTT-IPW; 2nd iteration", None, "./input_data/Q2JTT_IPW.PAML.txt"),
        # Varying dataset size
        ("Cherry; 1/2 data", None, "./input_data/Q1div2.PAML.txt"),
        ("Cherry; 1/4 data", None, "./input_data/Q1div4.PAML.txt"),
        ("Cherry; 1/8 data", None, "./input_data/Q1div8.PAML.txt"),
        ("Cherry; 1/16 data", None, "./input_data/Q1div16.PAML.txt"),
        ("Cherry; 1/128 data", None, "./input_data/Q1div128.PAML.txt"),
        # ("Cherry 1/1024 data", None, "./input_data/Q1div1024.PAML.txt"),
        # ("Cherry 1/2 data; 2nd iteration", None, "./input_data/Q2div2.PAML.txt"),
        # ("Cherry 1/4 data; 2nd iteration", None, "./input_data/Q2div4.PAML.txt"),
        # ("Cherry 1/8 data; 2nd iteration", None, "./input_data/Q2div8.PAML.txt"),
        # ("Cherry 1/16 data; 2nd iteration", None, "./input_data/Q2div16.PAML.txt"),
        # ("Cherry 1/128 data; 2nd iteration", None, "./input_data/Q2div128.PAML.txt"),
        # ("Cherry 1/1024 data; 2nd iteration", None, "./input_data/Q2div1024.PAML.txt"),
        #   ==> PhyML fails on one family for some reason...
        # ("EQU", None, "./input_data/synthetic_rate_matrices/PAML/EQU.PAML.txt"),
        # TODO: See how more iterations do.
        # ("Cherry; 3rd iteration", None, "./input_data/Q3v2.PAML.txt"),
        # ("Cherry; 4th iteration", None, "./input_data/Q4v2.PAML.txt"),
        # ("Cherry; 5th iteration", None, "./input_data/Q5v2.PAML.txt"),
        #   ==> PhyML fails on one family for some reason...
        # ("Cherry; 2nd iteration pytorh initJTTIPW", None, "./input_data/Q2.PAML.txt"),
        # ("Cherry_mixed", None,
        # "./input_data/synthetic_rate_matrices/PAML"
        # "/Q_learnt_from_LG_no_site_rates.PAML.txt"),
        # Testing models
        # ("WAG_PAML", None,
        #  "./input_data/synthetic_rate_matrices/PAML/WAG.PAML.txt"),
        # ("LG_PAML", None,
        #  "./input_data/synthetic_rate_matrices/PAML/WAG.PAML.txt"),
    ]


def get_subset_of_registered_models(
    model_names: List[str],
) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    Get a subset of the registered models.

    Args:
        model_names: The registered models with names in this list will be
            returned.

    Raises:
        ValueError if some model name in model_names is not registered.
    """
    assert model_names is not None
    registered_models = get_registered_models()
    models = [x for x in registered_models if x[0] in model_names]
    known_model_names = [x[0] for x in registered_models]
    for model_name in model_names:
        if model_name not in known_model_names:
            raise ValueError(f"Unknown model_name = {model_name}")
    return models


def wget_tarred_data_and_chmod(
    url: str,
    destination_directory: str,
    expected_number_of_files: int,
    mode: str = "555",
) -> None:
    """
    Download tar data from a url if not already present.

    Gets tarred data from `url` into `destination_directory` and chmods the
    data to 555 (or the `mode` specified) so that it is write protected.
    `expected_number_of_files` is the expected number of files after untarring.
    If the data is already present (which is determined by seeing whether the
    expected_number_of_files match), then the data is not downloaded again.

    Args:
        url: The url of the tar data.
        destination_directory: Where to untar the data to.
        expected_number_of_files: The expected number of files after
            untarring.
        mode: What mode to change the files to.

    Raises:
        Exception if the expected_number_of_files are not found after untarring,
            or if the data fails to download, etc.
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
            f"{url} has already been downloaded successfully to "
            f"{destination_directory}. Not downloading again."
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
) -> None:
    """
    Download the lg_TreeBase data.

    The data is hosted at:
    http://www.atgc-montpellier.fr/models/index.php?model=lg

    Args:
        destination_directory: Where to download the data to.
    """
    wget_tarred_data_and_chmod(
        url="http://www.atgc-montpellier.fr/download/datasets/models"
        "/lg_TreeBase.tar.gz",
        destination_directory=destination_directory,
        expected_number_of_files=59,
        mode="555",
    )


def get_lg_PfamTestingAlignments_data(
    destination_directory: str,
) -> None:
    """
    Download the lg_PfamTestingAlignments data

    The data is hosted at:
    http://www.atgc-montpellier.fr/models/index.php?model=lg

    Args:
        destination_directory: Where to download the data to.
    """
    wget_tarred_data_and_chmod(
        url="http://www.atgc-montpellier.fr/download/datasets/models"
        "/lg_PfamTestingAlignments.tar.gz",
        destination_directory=destination_directory,
        expected_number_of_files=500,
        mode="555",
    )


def _convert_lg_data(
    lg_training_data_dir: str,
    destination_directory: str,
) -> None:
    """
    Convert the LG MSAs from the PHYLIP format to our training format.

    Args:
        lg_training_data_dir: Where the MSAs in PHYLIP format are.
        destination_directory: Where to write the converted MSAs to.
    """
    logger.info("Converting LG Training data to our MSA training format ...")
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    protein_family_names = sorted(list(os.listdir(lg_training_data_dir)))
    for protein_family_name in protein_family_names:
        with open(
            os.path.join(lg_training_data_dir, protein_family_name), "r"
        ) as file:
            res = ""
            lines = file.read().split("\n")
            n_seqs, n_sites = map(int, lines[0].split(" "))
            for i in range(n_seqs):
                line = lines[2 + i]
                try:
                    protein_name, protein_sequence = line.split()
                except Exception:
                    raise ValueError(
                        f"For protein family {protein_family_name} , could "
                        f"not split line: {line}"
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
) -> None:
    """
    Get the lg_PfamTrainingAlignments.

    Downloads the lg_PfamTrainingAlignments data to the specified
    `destination_directory`, *converting it to our training MSA format in the
    process*.

    The data is hosted at:
    http://www.atgc-montpellier.fr/models/index.php?model=lg

    Args:
        destination_directory: Where to store the (converted) MSAs.
    """
    url = (
        "http://www.atgc-montpellier.fr/download/datasets/models"
        "/lg_PfamTrainingAlignments.tar.gz"
    )
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
            f"{url} has already been downloaded successfully "
            f"to {destination_directory}. Not downloading again."
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
            lg_training_data_dir=os.path.join(
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
    Run PHYML on an MSA with a given model.

    Run PHYML on the MSA in PHYLIP format specified by the `input_msa_path`,
    using either the given `model` or the PAML file specified by
    `rate_matrix_path` (exactly one of them should be specified), and
    with the given number of Gamma discrete rate categories
    `num_rate_categories`, the given `random_seed`, and `optimize_rates`.
    Returns the contents of the PHYML stats file, and of the PHYML site
    log-likelihoods. This function is wrapped with caching.

    Args:
        input_msa_path: Where the MSA in PHYLIP format is.
        model: E.g. 'LG'.
        rate_matrix_path: PAML file to the rate matrix to use.
        num_rate_categories: Number of discrete Gamma rate categories to use in
            PhyML.
        random_seed: The PhyML random seed.
        optimize_rates: Whether to optimize rates in PhyML.
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


def get_reported_results_df(pfam_or_treebase: str):
    """
    Gets the results table of the LG paper.

    The data is hosted at:
    http://www.atgc-montpellier.fr/models/index.php?model=lg

    Args:
        pfam_or_treebase: 'pfam' or 'treebase'.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if pfam_or_treebase == "treebase":
        df = pd.read_csv(os.path.join(dir_path, "Treebase.txt"), sep="\t")
    elif pfam_or_treebase == "pfam":
        df = pd.read_csv(os.path.join(dir_path, "Pfam.txt"), sep="\t")
    else:
        raise ValueError(
            f"pfam_or_treebase should be either 'pfam' or "
            f"'treebase'. You provided: {pfam_or_treebase}"
        )
    df = df.drop(0)
    df.set_index(["Name"], inplace=True)
    return df


@caching.cached()
def get_phyml_ll(
    pfam_or_treebase: str,
    input_msa_path: str,
    model: Optional[str],
    rate_matrix_path: Optional[str],
    num_rate_categories: int,
    random_seed: int,
    optimize_rates: bool,
) -> float:
    """
    Get the PhyML log-likelihood, either reported or de-novo.

    This function is a thin wrapper on top of run_phyml_cached that will
    check whether the `model` starts with the prefix 'r__'. If it does,
    (e.g. 'r__LG'), it will return the result reported in the LG paper
    for that model. Else, it will forward the call to run_phyml_cached.

    Args:
        pfam_or_treebase: 'pfam' or 'treebase'.
        input_msa_path: Where the MSA in PHYLIP format is.
        model: E.g. 'LG'.
        rate_matrix_path: PAML file to the rate matrix to use.
        num_rate_categories: Number of discrete Gamma rate categories to use in
            PhyML.
        random_seed: The PhyML random seed.
        optimize_rates: Whether to optimize rates in PhyML.
    """
    if model is not None and model.startswith("r__"):
        model_name = model.split("__")[-1]
        df = get_reported_results_df(pfam_or_treebase=pfam_or_treebase)
        protein_family_name = input_msa_path.split("/")[-1].split(".")[0]
        return df.loc[protein_family_name, model_name]
    else:
        phyml_stats, _ = run_phyml_cached(
            input_msa_path=input_msa_path,
            model=model,
            rate_matrix_path=rate_matrix_path,
            num_rate_categories=num_rate_categories,
            random_seed=random_seed,
            optimize_rates=optimize_rates,
        )
        return get_phyml_ll_from_phyml_stats(phyml_stats)


def get_filenames_Treebase() -> List[str]:
    """
    Get the list of filenames of the Treebase test data.

    The list is sorted from smallest MSA to largest (basically based on Taxa
    and Sites), making it easy to subset the smaller families for testing
    purposes.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(dir_path, "Treebase.txt"), sep="\t")
    df = df.drop(0)
    filenames = list(df.sort_values(by=["Tax", "Sites"]).Name)
    # Put two families with massive number of sites at the end.
    filenames = [x for x in filenames if x not in ["M2577", "M964"]] + [
        "M2577",
        "M964",
    ]
    filenames = [x + ".clean.ProtGB_phyml" for x in filenames]
    return filenames


def get_filenames_Pfam() -> List[str]:
    """
    Get the list of filenames of the Pfam test data.

    The list is sorted from smallest MSA to largest based on Taxa
    and Sites, making it easy to subset the smaller families for testing
    purposes.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(dir_path, "Pfam.txt"), sep="\t")
    df = df.drop(0)
    filenames = list(df.sort_values(by=["Tax", "Sites"]).Name)
    filenames = [x + ".txt-gb_phyml" for x in filenames]
    return filenames


def reproduce_JTT_WAG_LG_table(
    pfam_or_treebase: str,
    a3m_phylip_dir: str,
    model_names: Optional[List[str]] = None,
    num_rate_categories: int = 4,
    random_seed: int = 0,
    optimize_rates: bool = True,
    max_families: int = 100000000,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    For either Pfam or Treebase (specified via `pfam_or_treebase` argument),
    reproduce and extend with more models the LG table of results provided at:
    http://www.atgc-montpellier.fr/models/index.php?model=lg

    Args:
        pfam_or_treebase: 'pfam' or 'treebase'.
        a3m_phylip_dir: Where the alignments have been downloaded
            (which can be done via the get_lg_TreeBase_data and
            get_lg_PfamTestingAlignments_data functions).
        model_names: What models to use. If None, will use all models
            registered.
        num_rate_categories: Number of discrete Gamma rate categories to use in
            PhyML.
        random_seed: The PhyML random seed.
        optimize_rates: Whether to optimize rates in PhyML.
        max_families: How many families to reproduce the results on. This is
            useful for testing purposes, e.g. setting max_families=1.
        verbose: Verbosity level.
    """
    if pfam_or_treebase == "pfam":
        filenames = get_filenames_Pfam()
    elif pfam_or_treebase == "treebase":
        filenames = get_filenames_Treebase()

    rows = []
    if model_names is None:
        model_names = [x[0] for x in get_registered_models()]
    models = get_subset_of_registered_models(model_names=model_names)
    for filename in filenames[:max_families]:
        protein_family_name = filename.split(".")[0]
        input_msa_path = os.path.join(a3m_phylip_dir, filename)
        row = {}
        for (model_name, model, rate_matrix_path) in models:
            if verbose:
                print(f"Processing: {filename} with {model}")
            row[model_name] = get_phyml_ll(
                pfam_or_treebase=pfam_or_treebase,
                input_msa_path=input_msa_path,
                model=model,
                rate_matrix_path=rate_matrix_path,
                num_rate_categories=num_rate_categories,
                random_seed=random_seed,
                optimize_rates=optimize_rates,
            )
            row["Name"] = protein_family_name
        rows.append(row)
    res = pd.DataFrame(
        rows,
        columns=[
            "Name",
        ]
        + model_names,
    )
    res.set_index("Name", inplace=True)
    # Need to add the Tax and Sites columns
    df_reported = get_reported_results_df(pfam_or_treebase=pfam_or_treebase)
    res = res.merge(
        df_reported[["Tax", "Sites"]],
        left_index=True,
        right_index=True,
        how="left",
    )
    res = res[["Tax", "Sites"] + model_names]
    return res


def _get_phyml_ll(args: List) -> float:
    """
    Multiprocessing wrapper for get_phyml_ll
    """
    pfam_or_treebase = args[0]
    input_msa_path = args[1]
    model = args[2]
    rate_matrix_path = args[3]
    num_rate_categories = args[4]
    random_seed = args[5]
    optimize_rates = args[6]
    return get_phyml_ll(
        pfam_or_treebase=pfam_or_treebase,
        input_msa_path=input_msa_path,
        model=model,
        rate_matrix_path=rate_matrix_path,
        num_rate_categories=num_rate_categories,
        random_seed=random_seed,
        optimize_rates=optimize_rates,
    )


def reproduce_JTT_WAG_LG_table_parallel(
    pfam_or_treebase: str,
    a3m_phylip_dir: str,
    model_names: Optional[List[str]] = None,
    num_rate_categories: int = 4,
    random_seed: int = 0,
    optimize_rates: bool = True,
    max_families: int = 100000000,
    verbose: bool = False,
    n_process: int = 32,
) -> pd.DataFrame:
    """
    Parallel version of reproduce_JTT_WAG_LG_table

    Args:
        pfam_or_treebase: 'pfam' or 'treebase'.
        a3m_phylip_dir: Where the alignments have been downloaded
            (which can be done via the get_lg_TreeBase_data and
            get_lg_PfamTestingAlignments_data functions).
        model_names: What models to use. If None, will use all models
            registered.
        num_rate_categories: Number of discrete Gamma rate categories to use in
            PhyML.
        random_seed: The PhyML random seed.
        optimize_rates: Whether to optimize rates in PhyML.
        max_families: How many families to reproduce the results on. This is
            useful for testing purposes, e.g. setting max_families=1.
        verbose: Verbosity level.
        n_process: How many processes to use to parallelize computation.
    """
    if pfam_or_treebase == "pfam":
        filenames = get_filenames_Pfam()
    elif pfam_or_treebase == "treebase":
        filenames = get_filenames_Treebase()

    # First warm up the cache in parallel
    if model_names is None:
        model_names = [x[0] for x in get_registered_models()]
    models = get_subset_of_registered_models(model_names=model_names)
    map_args = []
    for filename in filenames[:max_families]:
        input_msa_path = os.path.join(a3m_phylip_dir, filename)
        for (_, model, rate_matrix_path) in models:
            map_args.append(
                (
                    pfam_or_treebase,
                    input_msa_path,
                    model,
                    rate_matrix_path,
                    num_rate_categories,
                    random_seed,
                    optimize_rates,
                )
            )
    map_func = _get_phyml_ll
    if n_process > 1:
        with multiprocessing.Pool(n_process) as pool:
            list(tqdm.tqdm(pool.imap(map_func, map_args), total=len(map_args)))
    else:
        list(tqdm.tqdm(map(map_func, map_args), total=len(map_args)))

    # Now get what we want
    return reproduce_JTT_WAG_LG_table(
        pfam_or_treebase=pfam_or_treebase,
        a3m_phylip_dir=a3m_phylip_dir,
        model_names=model_names,
        num_rate_categories=num_rate_categories,
        random_seed=random_seed,
        optimize_rates=optimize_rates,
        max_families=max_families,
        verbose=verbose,
    )


def reproduce_lg_paper_fig_4(
    pfam_or_treebase: str,
    a3m_phylip_dir: str,
    model_names: Optional[List[str]] = None,
    num_rate_categories: int = 4,
    random_seed: int = 0,
    optimize_rates: bool = True,
    max_families: int = 100000000,
    verbose: bool = False,
    n_process: int = 32,
    figsize: Tuple[float, float] = (6.4, 4.8),
    n_bootstraps: int = 0,
) -> Optional[pd.DataFrame]:
    """
    Reproduce Fig. 4 of the LG paper, adding the desired models.

    Args:
        pfam_or_treebase: 'pfam' or 'treebase'.
        a3m_phylip_dir: Where the alignments have been downloaded
            (which can be done via the get_lg_TreeBase_data and
            get_lg_PfamTestingAlignments_data functions).
        model_names: What models to use. If None, will use all models
            registered.
        num_rate_categories: Number of discrete Gamma rate categories to use in
            PhyML.
        random_seed: The PhyML random seed.
        optimize_rates: Whether to optimize rates in PhyML.
        max_families: How many families to reproduce the results on. This is
            useful for testing purposes, e.g. setting max_families=1.
        verbose: Verbosity level.
        n_process: How many processes to use to parallelize computation.
        figsize: The plot figure size.
        n_bootstraps: If >0, this number of test set bootstraps will be
            performed, and the resulting dataframe of size
            (n_bootstraps X |model_names|) will be returned.
    """
    if model_names is None:
        model_names = [x[0] for x in get_registered_models()]

    df = reproduce_JTT_WAG_LG_table_parallel(
        pfam_or_treebase=pfam_or_treebase,
        a3m_phylip_dir=a3m_phylip_dir,
        model_names=model_names,
        num_rate_categories=num_rate_categories,
        random_seed=random_seed,
        optimize_rates=optimize_rates,
        max_families=max_families,
        verbose=verbose,
        n_process=n_process,
    )

    def get_log_likelihoods(df: pd.DataFrame, model_names: List[str]):
        """
        Given a DataFrame like the LG results table, with Name as the index,
        returns the sum of log likelihoods for each model.
        """
        num_sites = df.Sites.sum()
        log_likelihoods = (
            2.0
            * (df[model_names].sum(axis=0) - df["JTT (reported)"].sum())
            / num_sites
        )
        return log_likelihoods

    y = get_log_likelihoods(df, model_names)
    yerr = None
    if n_bootstraps > 0:
        np.random.seed(0)
        y_bootstraps = []
        for _ in range(n_bootstraps):
            chosen_rows = np.random.choice(
                df.index,
                size=len(df.index),
                replace=True,
            )
            df_bootstrap = df.loc[chosen_rows]
            assert df_bootstrap.shape == df.shape
            y_bootstrap = get_log_likelihoods(df_bootstrap, model_names)
            y_bootstraps.append(y_bootstrap)
        y_bootstraps = np.array(y_bootstraps)
        assert y_bootstraps.shape == (n_bootstraps, len(model_names))
        # yerr = np.array(
        #     [
        #         [
        #             y[i] - np.quantile(y_bootstraps[:, i], 0.025),  # Below
        #             np.quantile(y_bootstraps[:, i], 0.975) - y[i],  # Above
        #         ]
        #         for i in range(len(model_names))
        #     ]
        # ).T

    colors = []
    for model_name in model_names:
        if "reported" in model_name:
            colors.append("black")
        elif "reproduced" in model_name:
            colors.append("blue")
        elif "Cherry" in model_name:
            colors.append("red")
        elif "MP" in model_name:
            colors.append("green")
        elif "JTT-IPW" in model_name:
            colors.append("grey")
        else:
            raise Exception(f"Unknown color for model: {model_name}")
    plt.figure(figsize=figsize)
    plt.title(pfam_or_treebase)
    plt.bar(x=model_names, height=y, color=colors, yerr=yerr)
    plt.xticks(rotation=270)
    ax = plt.gca()
    ax.yaxis.grid()
    plt.legend(
        handles=[
            mpatches.Patch(color="black", label="Reported"),
            mpatches.Patch(color="blue", label="Reproduced"),
            mpatches.Patch(color="red", label="Cherry"),
            mpatches.Patch(color="green", label="M. Parsimony"),
            mpatches.Patch(color="grey", label="JTT-IPW"),
        ]
    )
    plt.show()

    if n_bootstraps:
        return pd.DataFrame(y_bootstraps, columns=model_names)
