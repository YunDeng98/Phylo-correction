import argparse
import os
import sys

import logging

import numpy as np
import pandas as pd

# from fast_tree_benchmark_plot import nseqs_x_nsites_results_table

parser = argparse.ArgumentParser(
    description="Estimate total runtime of FastTree on all MSAs."
)
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
    "--grid_path",
    type=str,
    help="Path of file where the grid was written",
    required=False,
    default="../fast_tree_benchmark_output/benchmark_results.csv",
)
parser.add_argument(
    "--outdir",
    type=str,
    help="Path of directory where to write ouputs.",
    required=False,
    default="../fast_tree_benchmark_extrapolation_output",
)


def init_logger(outdir: str):
    logger = logging.getLogger("phylo_correction.phylogeny_generation")
    logger.setLevel(logging.DEBUG)
    # fmt_str = "[%(asctime)s] %(levelname)s @ line %(lineno)d: %(message)s"
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler(
        f"{outdir}/fast_tree_benchmark_extrapolation.log"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)


if __name__ == "__main__":
    # Pull out arguments
    args = parser.parse_args()
    a3m_dir = args.a3m_dir
    MAX_SEQS = args.max_seqs
    MAX_SITES = args.max_sites
    outdir = args.outdir
    grid_path = args.grid_path
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    init_logger(outdir=outdir)
    logger = logging.getLogger("phylo_correction.phylogeny_generation")

    if not os.path.exists(grid_path):
        raise ValueError(f"Could not find grid_path {grid_path}")
    nseqs_x_nsites_results_table = pd.read_csv(grid_path, index_col=0)

    if not os.path.exists(a3m_dir):
        raise ValueError(f"Could not find a3m_dir {a3m_dir}")
    times = []
    for i, filename in enumerate(os.listdir(a3m_dir)):
        if i % 100 == 0:
            print(f"Processed {i} files")
        filepath = os.path.join(a3m_dir, filename)
        with open(filepath, "r") as f:
            seqs = list(f)
            seed_protein = seqs[1].strip()
            assert all([not c.islower() for c in seed_protein])
            nsites = len(seqs[1].strip())
            assert len(seqs) % 2 == 0
            nseqs_idx = int(np.log2(min(len(seqs) / 2, MAX_SEQS) - 1))
            nsites_idx = int(np.log2(min(nsites, MAX_SITES) - 1))
            time = nseqs_x_nsites_results_table.iloc[nseqs_idx][nsites_idx]
            times.append(time)
        # if i == 102:
        #     break
    logger.info(
        f"Total time in seconds for building all trees in {a3m_dir} "
        f"with FastTree using at most MAX_SEQS={MAX_SEQS} and "
        f"MAX_SITES={MAX_SITES}: {sum(times)} (s)"
    )
    with open(os.path.join(outdir, "estimated_time.txt"), "w") as file:
        file.write(str(sum(times)))
        file.flush()
