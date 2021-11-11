import sys
import os
import time
import tempfile
import numpy as np
import pandas as pd
from typing import List, Tuple

from .MSA import MSA

import logging

from ete3 import Tree
from src.utils import verify_integrity

sys.path.append("../")
import Phylo_util


def copy_file_and_chmod(
    input_filepath: str,
    output_filepath: str,
):
    """
    Copy file and chmod to 555
    """
    input_filepath = os.path.abspath(input_filepath)
    output_filepath = os.path.abspath(output_filepath)
    if not os.path.exists(input_filepath):
        raise ValueError(f"Input file not exist at: {input_filepath}")
    verify_integrity(input_filepath)
    with open(output_filepath, "w") as output_file:
        with open(input_filepath, "r") as input_file:
            output_file.write(input_file.read())
            output_file.flush()
    os.system(f"chmod 555 {output_filepath}")


def get_rate_categories(
    tree_dir: str,
    protein_family_name: str,
    use_site_specific_rates: bool,
    L: int
) -> Tuple[List[float], List[int], List[int]]:
    """
    Returns the rates, site_cats, sites_kept for the given protein
    family.

    Args:
        tree_dir: Directory where the FastTree log is found.
        protein_family_name: Protein family name.
        use_site_specific_rates: If to use_site_specific_rates. If False, will
            just use a rate of 1 for all sites, which is the old behaviour.
            I.e. we are backwards compatible here.
        L: Number of sites. Used only if use_site_specific_rates=False.
    """
    outlog = os.path.join(tree_dir, protein_family_name + ".log")
    with open(outlog, "r") as outlog_file:
        lines = [line.strip().split() for line in outlog_file]
        # if line.startswith("NCategories") or line.startswith("Rates") or line.startswith("SiteCategories"):
        if not lines[0][0] == "NCategories":
            raise ValueError(f"NCategories not found in {outlog}")
        ncats = int(lines[0][1])
        if not lines[1][0] == "Rates":
            raise ValueError(f"Rates not found in {outlog}")
        if not len(lines[1][1:]) == ncats:
            raise ValueError(f"Rates should be {ncats} in length. Found {len(lines[1][1:])}.")
        rates = [float(x) for x in lines[1][1:]]
        if not lines[2][0] == "SiteCategories":
            raise ValueError(f"SiteCategories not found in {outlog}")
        site_cats = [int(x) - 1 for x in lines[2][1:]]  # FastTree uses 1-based indexing, so we shift by 1.
    out_sites_kept_path = os.path.join(tree_dir, protein_family_name + ".sites_kept")
    with open(out_sites_kept_path, "r") as out_sites_kept_file:
        sites_kept = [int(x) for x in out_sites_kept_file.read().strip().split()]
    if not len(site_cats) == len(sites_kept):
        raise ValueError(f"SiteCategories should have length {len(sites_kept)},"
                         f" but has {len(site_cats)} instead.")
    if not use_site_specific_rates:
        # Just force all rates to 1, and use all sites, which is the old
        # behavior. (So, we are backwards compatible.)
        rates = [1.0 for _ in rates]
        site_cats = [0] * L
        sites_kept = list(range(L))
    return rates, site_cats, sites_kept


def get_site_rate_from_site_id(
    site_id: int,
    rates: List[float],
    site_cats: List[int],
    sites_kept: List[int],
):
    """
    Precondition: the site_id must be in the sites_kept, or an error will be
    raised.
    """
    if not len(site_cats) == len(sites_kept):
        raise ValueError(f"site_cats and sites_kept should have the same length."
                         f" len(site_cats) = {len(site_cats)}; "
                         f"len(sites_kept) = {len(sites_kept)}")
    # Find its rate
    for site, site_cat in zip(sites_kept, site_cats):
        if site == site_id:
            return rates[site_cat]
    raise ValueError("Trying to retrieve rate for site that was not kept."
                     f" Site: {site_id}. sites_kept = {sites_kept}")


class PhylogenyGeneratorError(Exception):
    pass


def to_fast_tree_format(rate_matrix: np.array, output_path: str, pi: np.array):
    r"""
    The weird 20 x 21 format of FastTree, which is also column-stochastic.
    """
    amino_acids = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    rate_matrix_df = pd.DataFrame(rate_matrix, index=amino_acids, columns=amino_acids)
    rate_matrix_df = rate_matrix_df.transpose()
    rate_matrix_df['*'] = pi
    with open(output_path, "w") as outfile:
        for aa in amino_acids:
            outfile.write(aa + "\t")
        outfile.write("*\n")
        outfile.flush()
    rate_matrix_df.to_csv(output_path, sep="\t", header=False, mode='a')


def run_fast_tree_with_custom_rate_matrix(
    dir_path: str,
    rate_matrix: str,
    processed_msa_filename: str,
    outfile: str,
    outlog: str,
    fast_tree_cats: int,
) -> str:
    r"""
    This wrapper deals with the fact that FastTree only accepts normalized rate matrices
    as input. Therefore, to run FastTree with an arbitrary rate matrix, we first have
    to normalize it. After inference with FastTree, we have to 'de-normalize' the branch
    lengths to put them in the same time units as the original rate matrix.
    """
    with tempfile.NamedTemporaryFile("w") as scaled_tree_file:
        scaled_tree_filename = scaled_tree_file.name  # Where FastTree will write its output.
        with tempfile.NamedTemporaryFile("w") as scaled_rate_matrix_file:
            scaled_rate_matrix_filename = scaled_rate_matrix_file.name  # The rate matrix for FastTree
            Q_df = pd.read_csv(rate_matrix, sep="\t")
            if not (Q_df.shape == (20, 21)):
                raise ValueError(f"The rate matrix {rate_matrix} does not have dimension 20 x 21.")
            Q = np.array(Q_df.iloc[:20, :20].transpose())
            pi = np.array(Q_df.iloc[:20, 20]).reshape(1, 20)
            # Check that rows (originally columns) of Q add to 0
            if not np.sum(np.abs(Q.sum(axis=1))) < 0.01:
                raise ValueError(f"Custom rate matrix {rate_matrix} doesn't have columns that add up to 0.")
            # Check that the stationary distro is correct
            if not np.sum(np.abs(pi @ Q)) < 0.01:
                raise ValueError(f"Custom rate matrix {rate_matrix} doesn't have the stationary distribution.")
            # Compute the mutation rate.
            mutation_rate = pi @ -np.diag(Q)
            if abs(mutation_rate - 1.0) < 0.00001:
                # Can just use the original rate matrix
                command = f"{dir_path}/FastTree -quiet -trans {rate_matrix} -log {outlog} -cat {fast_tree_cats} < {processed_msa_filename} > {outfile}"
                os.system(command)
                return command
            # Normalize Q
            Q_normalized = Q / mutation_rate
            # Write out Q_normalized in FastTree format, for use in FastTree
            to_fast_tree_format(Q_normalized, output_path=scaled_rate_matrix_filename, pi=pi.reshape(20))
            # Run FastTree!
            command = f"{dir_path}/FastTree -quiet -trans {scaled_rate_matrix_filename} -log {outlog} -cat {fast_tree_cats} < {processed_msa_filename} > {scaled_tree_filename}"
            os.system(command)
            # De-normalize the branch lengths of the tree
            tree = Tree(scaled_tree_filename)

            def dfs_scale_tree(v: tree) -> None:
                for u in v.get_children():
                    u.dist = u.dist / mutation_rate
                    dfs_scale_tree(u)
            dfs_scale_tree(tree)
            tree.write(format=2, outfile=outfile)
            return command


def post_process_fast_tree_log(outlog: str):
    """
    We just want the sites and rates, so we prune the FastTree log file to keep
    just this information.
    """
    res = ""
    with open(outlog, "r") as infile:
        for line in infile:
            if line.startswith("NCategories") or line.startswith("Rates") or line.startswith("SiteCategories"):
                res += line
    with open(outlog, "w") as outfile:
        outfile.write(res)
        outfile.flush()


class FastTreePhylogeny:
    r"""
    Run FastTree on a given MSA file.

    Runs FastTree on the MSA file at f'{a3m_dir}/{protein_family_name}.a3m' and writes
    the output (a newick tree) to outdir/protein_family_name.newick. The MSA file
    is preprocessed with the MSA class (so: lowercase letters are ignored, and
    the sequences and sites are subsampled down to max_seqs and max_sites respectively).
    Uses the rate matrix at 'rate_matrix'. If 'rate_matrix' ends in 'None',
    the default FastTree rate matrix is used.

    Args:
        a3m_dir: Directory where the MSA file is found.
        protein_family_name: Name of the protein family.
        outdir: Directory where to write the output of FastTree - a .newick file.
        max_seqs: If nonzero, this number of sequences in the MSA file will be subsampled
            uniformly at random. The first sequence in the MSA file will always be sampled.
        max_sites: If nonzero, this number of sites in the MSA file will be subsampled
            uniformly at random.
        rate_matrix: Path to the rate matrix to use within FastTree. If ends in 'None', then
            the default rate matrix will be used in FastTree.
        fast_tree_cats: Number of Gamma rate categories to use in FastTree.
        use_cached: If True and the output file already exists, FastTree will NOT be run.

    Attributes:
        nseqs: Number of sequences in the MSA (after subsamping)
        nsites: Number of sites in the MSA (after subsampling)
        total_time: Total time taken to run FastTree on the MSA file.
    """
    def __init__(
        self,
        a3m_dir: str,
        protein_family_name: str,
        outdir: str,
        max_seqs: int,
        max_sites: int,
        rate_matrix: str,
        fast_tree_cats: int,
        use_cached: bool = False,
    ):
        logger = logging.getLogger("phylo_correction.FastTreePhylogeny")

        # Caching pattern: skip any computation as soon as possible
        outfile = os.path.join(outdir, protein_family_name) + ".newick"
        outlog = os.path.join(outdir, protein_family_name) + ".log"
        out_sites_kept = os.path.join(outdir, protein_family_name) + ".sites_kept"
        if use_cached and os.path.exists(outfile) and os.path.exists(outlog) and os.path.exists(out_sites_kept):
            verify_integrity(outfile)
            verify_integrity(outlog)
            verify_integrity(out_sites_kept)
            if len(open(outfile).read()) < 3:
                raise Exception(f"Tree at: {os.path.abspath(outfile)} is corrupted")
            # logger.info(f"Skipping. Cached FastTree output for family {protein_family_name} at {outfile}")
            return

        dir_path = os.path.dirname(os.path.realpath(__file__))
        fast_tree_bin_path = os.path.join(dir_path, 'FastTree')
        fast_tree_path = os.path.join(dir_path, 'FastTree.c')
        if not os.path.exists(fast_tree_bin_path):
            # TODO: Make this part of installation?
            logger.info("Getting FastTree.c file ...")
            os.system(
                f"wget http://www.microbesonline.org/fasttree/FastTree.c -P {dir_path}"
            )
            logger.info("Compiling FastTree ...")
            # See http://www.microbesonline.org/fasttree/#Install
            os.system(
                f"gcc -DNO_SSE -DUSE_DOUBLE -O3 -finline-functions -funroll-loops -Wall -o {fast_tree_bin_path} {fast_tree_path} -lm"
            )

        if not os.path.exists(outdir):
            logger.info(f"Creating outdir {outdir}")
            os.makedirs(outdir)

        # Read MSA into standardized format (lowercase amino acids are removed.)
        msa = MSA(
            a3m_dir=a3m_dir,
            protein_family_name=protein_family_name,
            max_seqs=max_seqs,
            max_sites=max_sites
        )
        # Write (standardized) MSA
        with tempfile.NamedTemporaryFile("w") as processed_msa_file:
            processed_msa_filename = processed_msa_file.name
            msa.write_to_file(processed_msa_filename)
            # Run FastTree on (preprocessed) MSA file
            with open(out_sites_kept, "w") as out_sites_kept_file:
                out_sites_kept_file.write(' '.join(map(str, msa.sites_kept)))
                out_sites_kept_file.flush()
            time_start = time.time()
            dir_path = os.path.dirname(os.path.realpath(__file__))
            if rate_matrix[-4:] == 'None':
                assert(not os.path.exists(rate_matrix))
                logger.info(f"Running FastTree with default rate matrix on MSA:\n{msa}")
                command = f"{dir_path}/FastTree -quiet -log {outlog} -cat {fast_tree_cats} < {processed_msa_filename} > {outfile}"
                os.system(command)
            else:
                if not os.path.exists(rate_matrix):
                    logger.error(f"Could not find rate matrix {rate_matrix}")
                    raise PhylogenyGeneratorError(f"Could not find rate matrix {rate_matrix}")
                logger.info(f"Running FastTree with rate matrix {rate_matrix} on MSA:\n{msa}")
                command = run_fast_tree_with_custom_rate_matrix(
                    dir_path=dir_path,
                    rate_matrix=rate_matrix,
                    processed_msa_filename=processed_msa_filename,
                    outfile=outfile,
                    outlog=outlog,
                    fast_tree_cats=fast_tree_cats,
                )
            time_end = time.time()
            self._total_time = time_end - time_start
            logger.info(f"Time taken: {self.total_time}")
            self._msa = msa
            self._protein_family_name = protein_family_name
            post_process_fast_tree_log(outlog)
            os.system(f"chmod 555 {outfile}")
            os.system(f"chmod 555 {outlog}")
            os.system(f"chmod 555 {out_sites_kept}")
            if len(open(outfile).read()) < 3:
                raise Exception(
                    f"Tree at: {os.path.abspath(outfile)} is corrupted."
                    f"Command was: {command} . Input MSA was:\n{open(processed_msa_filename).read()}")

    @property
    def total_time(self) -> float:
        r"""
        The total time it took to run FastTree.
        """
        return self._total_time

    @property
    def nseqs(self) -> int:
        r"""
        Number of sequences in the MSA.
        """
        return self._msa.nseqs

    @property
    def nsites(self) -> int:
        r"""
        Number of sites in the MSA.
        """
        return self._msa.nsites

    def __str__(self) -> str:
        res = (
            f"FastTreePhylogeny for {self._protein_family_name}\nTotal time: "
            f"{self.total_time} (s)\n"
        )
        res += str(self._msa)
        return res
