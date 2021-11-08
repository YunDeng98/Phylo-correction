r"""
http://biowiki.org/wiki/index.php/Xrate_Software

Reads MSAs and trees and converts them to stockholm format (the input format
for XRATE)
"""
import multiprocessing
import os
import sys
sys.path.append("../")
import Phylo_util

import logging
import numpy as np
import tqdm
from ete3 import Tree

from typing import List


from src.utils import subsample_protein_families, verify_integrity


def convert_parsimony_file_to_stock(
    protein_family_name: str,
    parsimony_input_path: str,
    tree_input_path: str,
) -> str:
    """
    Convert a parsimony alignment file to stockholm format.

    The only reason we read the maximum parsimony file is because it contains
    the subsampled MSA, and the filtered sites.

    XRATE requires inputs in the Stockholm format, which is why this method exists.
    """
    res = "# STOCKHOLM 1.0\n"

    def dfs_rename_nodes(v):
        v.name = protein_family_name + '-' + v.name
        for u in v.get_children():
            dfs_rename_nodes(u)
    tree = Tree(tree_input_path, format=3)
    dfs_rename_nodes(tree)
    res += "#=GF NH " + tree.write(format=3) + "\n"

    # Read MSA
    with open(parsimony_input_path) as file:
        lines = list(file)
        n_lines = len(lines)
        for i in range(0, n_lines):
            line = lines[i].split(' ')
            if not line[0].startswith('seq'):
                continue
            protein_name = line[0]
            protein_seq = line[1]
            protein_seq.replace('-', '.')
            res += protein_family_name + '-' + protein_name + " " + protein_seq
    return res


def map_func(args: List) -> None:
    parsimony_dir = args[0]
    protein_family_name = args[1]
    outdir = args[2]
    use_cached = args[3]

    logger = logging.getLogger("phylo_correction.xrate_input_generator")

    # Caching pattern: skip any computation as soon as possible
    transition_filename = os.path.join(outdir, protein_family_name + ".stock")
    if use_cached and os.path.exists(transition_filename):
        verify_integrity(transition_filename)
        # logger.info(f"Skipping. Cached transitions for family {protein_family_name} at {transition_filename}")
        return

    logger.info(f"Starting on family {protein_family_name}")

    res = convert_parsimony_file_to_stock(
        protein_family_name=protein_family_name,
        parsimony_input_path=os.path.join(parsimony_dir, f"{protein_family_name}.parsimony"),
        tree_input_path=os.path.join(parsimony_dir, f"{protein_family_name}.newick"),
    )

    stock_filename = os.path.join(outdir, protein_family_name + ".stock")
    with open(stock_filename, "w") as stock_file:
        stock_file.write(res)
        stock_file.flush()
    os.system(f"chmod 555 {stock_filename}")


class XRATEInputGenerator:
    r"""
    Generate input for XRATE, given the MSAs and trees.

    The hyperparameters are passed in '__init__', and the outputs are only
    computed upon call to the 'run' method.

    Args:
        a3m_dir_full: Directory with MSAs for ALL protein families. Used
            to determine which max_families will get subsampled.
        parsimony_dir: Directory where the MSA files (.parsimony) and
            trees (.newick) are found. Note that the ancestral states are
            ignored compeltely; however, the parsimony files contain the
            post-processed MSAs so we read from them.
        n_process: Number of processes used to parallelize computation.
        expected_number_of_MSAs: The number of files in a3m_dir. This argument
            is only used to sanity check that the correct a3m_dir is being used.
            It has no functional implications.
        outdir: Directory where the generated stockholm files (.stock files)
            will be found.
        max_families: Only run on 'max_families' randomly chosen files in a3m_dir_full.
            This is useful for testing and to see what happens if less data is used.
        use_cached: If True and the output file already exists for a family,
            all computation will be skipped for that family.
    """
    def __init__(
        self,
        a3m_dir_full: str,
        parsimony_dir: str,
        n_process: int,
        expected_number_of_MSAs: int,
        outdir: str,
        max_families: int,
        use_cached: bool = False,
    ):
        self.a3m_dir_full = a3m_dir_full
        self.parsimony_dir = parsimony_dir
        self.n_process = n_process
        self.expected_number_of_MSAs = expected_number_of_MSAs
        self.outdir = outdir
        self.max_families = max_families
        self.use_cached = use_cached

    def run(self) -> None:
        logger = logging.getLogger("phylo_correction.xrate_input_generator")
        logger.info(f"Starting on max_families={self.max_families}, outdir: {self.outdir}")

        a3m_dir_full = self.a3m_dir_full
        parsimony_dir = self.parsimony_dir
        n_process = self.n_process
        expected_number_of_MSAs = self.expected_number_of_MSAs
        outdir = self.outdir
        max_families = self.max_families
        use_cached = self.use_cached

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if not os.path.exists(a3m_dir_full):
            raise ValueError(f"Could not find a3m_dir_full {a3m_dir_full}")

        protein_family_names = subsample_protein_families(
            a3m_dir_full,
            expected_number_of_MSAs,
            max_families
        )

        # print(f"protein_family_names = {protein_family_names}")

        map_args = [
            [parsimony_dir, protein_family_name, outdir, use_cached]
            for protein_family_name in protein_family_names
        ]
        if n_process > 1:
            with multiprocessing.Pool(n_process) as pool:
                list(tqdm.tqdm(pool.imap(map_func, map_args), total=len(map_args)))
        else:
            list(tqdm.tqdm(map(map_func, map_args), total=len(map_args)))


def rate_matrix_to_grammar(Q: np.array) -> str:
    assert(Q.shape == (20, 20))
    res = """;; Grammar nullprot
;;
(grammar
 (name nullprot)
 (update-rates 1)
 (update-rules 1)

 ;; Transformation rules for grammar symbols

 ;; State Start
 ;;
 (transform (from (Start)) (to (S0)) (prob 0.5))
 (transform (from (Start)) (to ()) (prob 0.5))

 ;; State S0
 ;;
 (transform (from (S0)) (to (A0 S0*)) (gaps-ok)
  (minlen 1))
 (transform (from (S0*)) (to ()) (prob 0.5))
 (transform (from (S0*)) (to (S0)) (prob 0.5))

 ;; Markov chain substitution models

 (chain
  (update-policy rev)
  (terminal (A0))

  ;; initial probability distribution
"""
    pi = Phylo_util.solve_stationery_dist(Q)
    amino_acids = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    for i, aa in enumerate(amino_acids):
        res += f"  (initial (state ({aa.lower()})) (prob {pi[i]}))\n"
    res += "\n"
    res += "  ;; mutation rates\n"
    for i, aa1 in enumerate(amino_acids):
        for j, aa2 in enumerate(amino_acids):
            if i != j:
                res += f"  (mutate (from ({aa1.lower()})) (to ({aa2.lower()})) (rate {Q[i, j]}))\n"
    res += """ )  ;; end chain A0

)  ;; end grammar nullprot

;; Alphabet Protein
;;
(alphabet
 (name Protein)
 (token (a r n d c q e g h i l k m f p s t w y v))
 (extend (to x) (from a) (from r) (from n) (from d) (from c) (from q) (from e) (from g) (from h) (from i) (from l) (from k) (from m) (from f) (from p) (from s) (from t) (from w) (from y) (from v))
 (extend (to b) (from n) (from d))
 (extend (to z) (from q) (from e))
 (wildcard *)
)  ;; end alphabet Protein

"""
    return res
