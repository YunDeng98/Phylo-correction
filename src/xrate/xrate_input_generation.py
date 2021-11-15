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
import tempfile

from typing import List, Tuple


from src.utils import subsample_protein_families, verify_integrity
from src.phylogeny_generation.FastTreePhylogeny import get_rate_categories


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
            if len(line) == 1:
                assert(i == 0)
                continue
            if line[0].startswith('internal-'):
                continue
            protein_name = line[0]
            protein_seq = line[1]
            protein_seq.replace('-', '.')
            res += protein_family_name + '-' + protein_name + " " + protein_seq
    return res


def subset_sites(seq: str, sites_to_subset: List[int]):
    """
    Subset sites in a string. Used to subset the
    positions of an MSA corresponding to a rate category.
    """
    return ''.join([seq[i] for i in sites_to_subset])


def write_out_msa(
    msa: List[Tuple[str, str]],
    msa_filepath: str,
) -> None:
    """
    Write out an msa consisting of (protein_name, sequence) tuples.

    This is used to subset the positions of an MSA corresponding to a rate
    category. The MSA thus has the same format as the output of the
    maximum parsimony step.
    """
    res = f"{len(msa)}\n"
    for (protein_name, sequence) in msa:
        res += f"{protein_name} {sequence}\n"
    with open(msa_filepath, "w") as outfile:
        outfile.write(res)
        outfile.flush()


def convert_parsimony_file_to_one_stock_per_rate_category(
    protein_family_name: str,
    parsimony_dir: str,
    num_rate_categories: int,
) -> List[str]:
    """
    Converts the MSA for the given protein family into one .stock file per site
    rate category.

    The strategy is to just loop over each rate category and subset the
    positions with the given rate category in the MSA, scale the tree,
    then just call convert_parsimony_file_to_stock.
    """
    parsimony_input_path = os.path.join(parsimony_dir, f"{protein_family_name}.parsimony")
    tree_input_path = os.path.join(parsimony_dir, f"{protein_family_name}.newick")

    # Read MSA
    msa = []
    with open(parsimony_input_path) as file:
        lines = list(file)
        n_lines = len(lines)
        for i in range(0, n_lines):
            line = lines[i].split(' ')
            if len(line) == 1:
                assert(i == 0)
                continue
            assert(len(line) == 2)
            protein_name = line[0]
            protein_seq = line[1]
            msa.append((protein_name, protein_seq))
    L = len(msa[0][1])

    rates, site_cats, sites_kept = get_rate_categories(
        tree_dir=parsimony_dir,
        protein_family_name=protein_family_name,
        use_site_specific_rates=True,
        L=L,
    )

    if len(rates) != num_rate_categories:
        raise ValueError(
            "The number of rate categories obtained from FastTree log is: "
            f"{len(rates)}, but you were expecting to see "
            f"{num_rate_categories}."
        )

    assert(len(sites_kept) == len(site_cats))

    res = []
    for rate_category in range(num_rate_categories):
        # Subset the MSA for this rate category, and scale the tree.
        # Determine the sites for this rate category
        sites_for_rate_category = [sites_kept[i] for i in range(len(sites_kept)) if site_cats[i] == rate_category]
        msa_subset = [
            (
                protein_name,
                subset_sites(
                    seq=protein_seq,
                    sites_to_subset=sites_for_rate_category
                )
            )
            for (protein_name, protein_seq) in msa
        ]
        tree = Tree(tree_input_path, format=3)

        def dfs_scale_tree(v, scaling_factor: float):
            v.dist = v.dist * scaling_factor
            for u in v.get_children():
                dfs_scale_tree(u, scaling_factor=scaling_factor)
        tree = Tree(tree_input_path, format=3)
        dfs_scale_tree(tree, scaling_factor=rates[rate_category])
        # Now write out the MSA and the tree, and call
        # convert_parsimony_file_to_stock
        with tempfile.NamedTemporaryFile("w") as msa_subset_file:
            msa_subset_filename = msa_subset_file.name
            with tempfile.NamedTemporaryFile("w") as tree_file:
                tree_filename = tree_file.name
                write_out_msa(msa_subset, msa_subset_filename)
                tree.write(format=3, outfile=tree_filename)
                stock = convert_parsimony_file_to_stock(
                    protein_family_name=protein_family_name + f'__rc_{rate_category}',
                    parsimony_input_path=msa_subset_filename,
                    tree_input_path=tree_filename,
                )
                res.append(stock)
    if len(res) == 0:
        raise Exception("It appears like no rate category contained any "
                        "sites. This is not possible!")
    return res


def get_stock_path_for_rate_category(
    dirname: str,
    protein_family_name: str,
    rate_category: int
):
    """
    The path to the stockholm filepath for `protein_family_name`
    and rate category `rate_category` under dirname.
    """
    return os.path.join(dirname, f"{protein_family_name}__rc_{rate_category}.stock")


def get_stock_filenames(
    stock_dir: str,
    protein_family_names: List[str],
    use_site_specific_rates: bool,
    num_rate_categories: int,
) -> List[str]:
    """
    Logic to get the stock files. This logic depends on whether we
    use_site_specific_rates
    """
    if not use_site_specific_rates:
        stock_input_paths=[
            os.path.join(stock_dir, f"{protein_family_name}.stock")
            for protein_family_name in protein_family_names
        ]
    else:
        stock_input_paths=[
            get_stock_path_for_rate_category(
                stock_dir,
                protein_family_name,
                i
            )
            for protein_family_name in protein_family_names for i in range(num_rate_categories)
        ]
    return stock_input_paths


def map_func(args: List) -> None:
    parsimony_dir = args[0]
    protein_family_name = args[1]
    outdir = args[2]
    use_cached = args[3]
    use_site_specific_rates = args[4]
    num_rate_categories = args[5]

    logger = logging.getLogger("phylo_correction.xrate_input_generator")

    # Caching pattern: skip any computation as soon as possible
    stock_filenames = get_stock_filenames(
        stock_dir=outdir,
        protein_family_names=[protein_family_name],
        use_site_specific_rates=use_site_specific_rates,
        num_rate_categories=num_rate_categories,
    )
    if use_cached and all([os.path.exists(stock_filename) for stock_filename in stock_filenames]):
        [verify_integrity(stock_filename) for stock_filename in stock_filenames]
        return

    logger.info(f"Starting on family {protein_family_name}")
    if not use_site_specific_rates:
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
    else:
        # Here I now have to figure out how to deal with the use of
        # site-specific rates.
        # I should do something similar to transition extraction to get the site
        # rates.
        res = convert_parsimony_file_to_one_stock_per_rate_category(
            protein_family_name=protein_family_name,
            parsimony_dir=parsimony_dir,
            num_rate_categories=num_rate_categories,
        )
        if len(res) != num_rate_categories:
            raise Exception(f"Expected {num_rate_categories} stock MSAs for family {protein_family_name}, but got {len(res)}.")

        stock_filenames = get_stock_filenames(
            stock_dir=outdir,
            protein_family_names=[protein_family_name],
            use_site_specific_rates=use_site_specific_rates,
            num_rate_categories=num_rate_categories,
        )
        for i in range(num_rate_categories):
            with open(stock_filenames[i], "w") as stock_file:
                stock_file.write(res[i])
                stock_file.flush()
            os.system(f"chmod 555 {stock_filenames[i]}")


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
            ignored completely; however, the parsimony files contain the
            post-processed MSAs (e.g. removal of lowercase amino acids)
            so we read from them. Also, the site rates and sites kept
            files are here. (Just like with the transition extraction
            steps, this is done to keep the dependency chain linear.)
        n_process: Number of processes used to parallelize computation.
        expected_number_of_MSAs: The number of files in a3m_dir. This argument
            is only used to sanity check that the correct a3m_dir is being used.
            It has no functional implications.
        outdir: Directory where the generated stockholm files (.stock files)
            will be found.
        max_families: Only run on 'max_families' randomly chosen files in a3m_dir_full.
            This is useful for testing and to see what happens if less data is used.
        use_site_specific_rates: Whether to use site specific rates. When True,
            we get the LG method; when False, we get the WAG method.
        num_rate_categories: The number of rate categories, in case they shall
            be used (when use_site_specific_rates=True).
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
        use_site_specific_rates: bool,
        num_rate_categories: int,
        use_cached: bool = False,
    ):
        self.a3m_dir_full = a3m_dir_full
        self.parsimony_dir = parsimony_dir
        self.n_process = n_process
        self.expected_number_of_MSAs = expected_number_of_MSAs
        self.outdir = outdir
        self.max_families = max_families
        self.use_site_specific_rates = use_site_specific_rates
        self.num_rate_categories = num_rate_categories
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
        use_site_specific_rates = self.use_site_specific_rates
        num_rate_categories = self.num_rate_categories
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
            [parsimony_dir, protein_family_name, outdir, use_cached, use_site_specific_rates, num_rate_categories]
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
