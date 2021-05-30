import argparse
import logging
import multiprocessing
import os
import sys
import tqdm
import hashlib
import numpy as np
import random

from .ContactMatrix import ContactMatrix


parser = argparse.ArgumentParser(
    description="Generate contacts for all protein families."
)
parser.add_argument(
    "--a3m_dir",
    type=str,
    help="Directory where the MSAs are found (.a3m files)",
    required=True,
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

    logger = logging.getLogger("contact_generation")
    seed = int(hashlib.md5((protein_family_name + 'contact_generation').encode()).hexdigest()[:8], 16)
    logger.info(f"Setting random seed to: {seed}")
    np.random.seed(seed)
    random.seed(seed)

    contact_matrix = ContactMatrix(
        pdb_dir=pdb_dir,
        protein_family_name=protein_family_name,
        armstrong_cutoff=armstrong_cutoff,
    )
    outfile = os.path.join(outdir, protein_family_name + ".cm")
    contact_matrix.write_to_file(outfile)


class ContactGenerator:
    def __init__(
        self,
        a3m_dir: str,
        pdb_dir: str,
        armstrong_cutoff: float,
        n_process: int,
        expected_number_of_families: int,
        outdir: str,
        max_families: int,
    ):
        self.a3m_dir = a3m_dir
        self.pdb_dir = pdb_dir
        self.armstrong_cutoff = armstrong_cutoff
        self.n_process = n_process
        self.expected_number_of_families = expected_number_of_families
        self.outdir = outdir
        self.max_families = max_families

    def run(self):
        a3m_dir = self.a3m_dir
        pdb_dir = self.pdb_dir
        armstrong_cutoff = self.armstrong_cutoff
        n_process = self.n_process
        expected_number_of_families = self.expected_number_of_families
        outdir = self.outdir
        max_families = self.max_families

        init_logger()

        if os.path.exists(outdir):
            raise ValueError(
                f"outdir {outdir} already exists. Aborting not to " f"overwrite!"
            )
        os.makedirs(outdir)

        if not os.path.exists(pdb_dir):
            raise ValueError(f"Could not find pdb_dir {pdb_dir}")

        if not os.path.exists(a3m_dir):
            raise ValueError(f"Could not find pdb_dir {a3m_dir}")

        filenames = list(os.listdir(a3m_dir))
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


def _main():
    # Pull out arguments
    args = parser.parse_args()
    contact_generator = ContactGenerator(
        a3m_dir=args.a3m_dir,
        pdb_dir=args.pdb_dir,
        armstrong_cutoff=args.armstrong_cutoff,
        n_process=args.n_process,
        expected_number_of_families=args.expected_number_of_families,
        outdir=args.outdir,
        max_families=args.max_families,
    )
    contact_generator.run()


if __name__ == "__main__":
    _main()
