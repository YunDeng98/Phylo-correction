import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from MSA import MSA
from FastTreePhylogeny import FastTreePhylogeny

import logging

parser = argparse.ArgumentParser(description="Benchmark FastTree.")
parser.add_argument(
    "--a3m_dir",
    type=str,
    help="Directory where the MSAs are found (.a3m files)",
    required=False,
    default="../a3m",
)
parser.add_argument(
    "--max_seqs",
    type=int,
    help="Maximum number of sequences to use per family",
    required=True,
)
parser.add_argument(
    "--max_sites",
    type=int,
    help="Maximum number of sites to use per family",
    required=True,
)
parser.add_argument(
    "--protein_family_name",
    type=str,
    help="Protein family to use for the benchmark",
    required=False,
    default="1twf_1_B",
)
parser.add_argument(
    "--outdir",
    type=str,
    help="Path of directory where to write ouputs.",
    required=False,
    default="../fast_tree_benchmark_output",
)


def init_logger(outdir: str):
    logger = logging.getLogger("phylogeny_generation")
    logger.setLevel(logging.DEBUG)
    # fmt_str = "[%(asctime)s] %(levelname)s @ line %(lineno)d: %(message)s"
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler(f"{outdir}/fast_tree_benchmark.log")
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)


if __name__ == "__main__":
    # Pull out arguments
    args = parser.parse_args()
    a3m_dir = args.a3m_dir
    protein_family_name = args.protein_family_name
    outdir = args.outdir

    if os.path.exists(outdir):
        raise ValueError(
            f"Output directory {outdir} already exists. It would be "
            f"overwritten. Aborting!"
        )

    os.makedirs(outdir)
    init_logger(outdir)

    logger = logging.getLogger("phylogeny_generation")
    np.random.seed(1)
    max_seqs_grid = [2 ** i for i in range(1, 30) if 2 ** i <= args.max_seqs]
    max_sites_grid = [2 ** i for i in range(1, 30) if 2 ** i <= args.max_sites]
    logger.info(
        f"Benchmarking FastTree's speed.\n"
        f"Trying these number of sequences:\n"
        f"{max_seqs_grid}.\n"
        f"Trying these numbers of sites:\n"
        f"{max_sites_grid}."
    )

    # First check that the maximum values can be realized by the chosen
    # protein family, or else the benchmark will not be reliable.
    msa = MSA(
        a3m_dir,
        protein_family_name,
        max_seqs=max(max_seqs_grid),
        max_sites=max(max_sites_grid),
    )
    if msa.nseqs < max(max_seqs_grid):
        raise ValueError(
            f"MSA for family {protein_family_name} contains {msa.nseqs} "
            f"sequences, which is fewer than "
            f"{max(max_seqs_grid)} sequences. Cannot perform the desired "
            f"benchmark! Shrink the grid, or choose a different protein "
            f"family."
        )
    if msa.nsites < max(max_sites_grid):
        raise ValueError(
            f"MSA for family {protein_family_name} contains {msa.nsites} sites,"
            f" which is fewer than "
            f"{max(max_sites_grid)} sites. Cannot perform the desired "
            f"benchmark! Shrink the grid, or choose a different protein "
            "family."
        )

    # Now benchmark!
    res = np.zeros(shape=(len(max_seqs_grid), len(max_sites_grid)))
    for i, max_seqs in enumerate(max_seqs_grid):
        for j, max_sites in enumerate(max_sites_grid):
            tree = FastTreePhylogeny(
                a3m_dir=a3m_dir,
                protein_family_name=protein_family_name,
                outdir=outdir,
                max_seqs=max_seqs,
                max_sites=max_sites,
            )
            res[i, j] = tree.total_time

    # Write out results
    res = pd.DataFrame(res, index=max_seqs_grid, columns=max_sites_grid)
    logger.info(
        f"Writing out benchmarking results to {outdir}/benchmark_results.csv"
    )
    res.to_csv(os.path.join(outdir, "benchmark_results.csv"))

    logger.info(
        f"Plotting benchmarking results to "
        f"{outdir}/fast_tree_benchmark_plot.png"
    )
    sns.heatmap(
        res,
        yticklabels=max_seqs_grid,
        xticklabels=max_sites_grid,
        annot=True,
        fmt=".0f",
    )
    plt.ylabel("Number of sequences")
    plt.xlabel("Number of sites")
    plt.title("FastTree runtime (in seconds)")
    plt.savefig(f"{outdir}/fast_tree_benchmark_plot.png")
