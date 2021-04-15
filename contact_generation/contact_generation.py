import argparse
import logging
import multiprocessing
import os
import sys
import tqdm

from ContactMatrix import ContaxtMatrix


parser = argparse.ArgumentParser(
    description="Generate contacts for all protein families."
)
parser.add_argument(
    "--pdb_dir",
    type=str,
    help="Directory where the PDB files are found (.pdb files)",
    required=True,
)
parser.add_argument(
    "--outdir",
    type=str,
    help="Directory where the contact matrices will be found.",
    required=True,
)
parser.add_argument(
    "--armstrong_cutoff",
    type=float,
    help="Armstrong cutoff to use",
    required=True,
)
parser.add_argument(
    "--n_process",
    type=int,
    help="Number of processes to use",
    required=True,
)
parser.add_argument(
    "--expected_number_of_families",
    type=int,
    help="Expected number of families",
    required=True,
)
parser.add_argument(
    "--max_families",
    type=int,
    help="Maximum number of families to run on.",
    required=False,
    default=100000000,
)


def init_logger():
    logger = logging.getLogger("contact_generation")
    logger.setLevel(logging.DEBUG)
    # fmt_str = "[%(asctime)s] %(levelname)s @ line %(lineno)d: %(message)s"
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler("contact_generation.log")
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)


def map_func(args):
    pdb_dir = args[0]
    protein_family_name = args[1]
    outdir = args[2]
    armstrong_cutoff = args[3]
    contact_matrix = ContaxtMatrix(
        pdb_dir=pdb_dir,
        protein_family_name=protein_family_name,
        armstrong_cutoff=armstrong_cutoff,
    )
    outfile = os.path.join(outdir, protein_family_name + ".cm")
    contact_matrix.write_to_file(outfile)


if __name__ == "__main__":
    # Pull out arguments
    args = parser.parse_args()
    pdb_dir = args.pdb_dir
    armstrong_cutoff = args.armstrong_cutoff
    n_process = args.n_process
    expected_number_of_families = args.expected_number_of_families
    outdir = args.outdir
    max_families = args.max_families

    init_logger()

    if os.path.exists(outdir):
        raise ValueError(
            f"outdir {outdir} already exists. Aborting not to " f"overwrite!"
        )
    os.mkdir(outdir)

    if not os.path.exists(pdb_dir):
        raise ValueError(f"Could not find pdb_dir {pdb_dir}")

    filenames = list(os.listdir(pdb_dir))
    if not len(filenames) == expected_number_of_families:
        raise ValueError(
            f"Number of families is {len(filenames)}, does not match "
            f"expected {expected_number_of_families}"
        )
    protein_family_names = [x.split(".")[0] for x in filenames][:max_families]

    map_args = [
        (pdb_dir, protein_family_name, outdir, armstrong_cutoff)
        for protein_family_name in protein_family_names
    ]
    with multiprocessing.Pool(n_process) as pool:
        list(tqdm.tqdm(pool.imap(map_func, map_args), total=len(map_args)))
