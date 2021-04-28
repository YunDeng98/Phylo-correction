r"""
Runs maximum parsimony reconstruction for a given protein family.
Uses call to C++ implementation.
"""
import argparse
import multiprocessing
import os
import sys

import logging
import numpy as np
import tempfile
import tqdm
from typing import Dict, Tuple

from ete3 import Tree

sys.path.append('../')

from phylogeny_generation.MSA import MSA


def init_logger():
    logger = logging.getLogger("maximum_parsimony")
    logger.setLevel(logging.DEBUG)
    # fmt_str = "[%(asctime)s] %(levelname)s @ line %(lineno)d: %(message)s"
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler("maximum_parsimony.log")
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)


parser = argparse.ArgumentParser(description="Reconstruct ancestral states with maximum parsimony for the given protein family.")
parser.add_argument(
    "--a3m_dir",
    type=str,
    help="Directory where the MSAs are found (.a3m files)",
    required=True,
)
parser.add_argument(
    "--tree_dir",
    type=str,
    help="Directory where the trees are found (.newick files)",
    required=True,
)
parser.add_argument(
    "--outdir",
    type=str,
    help="Directory where the maximum parsimony reconstruction will be found.",
    required=True,
)
parser.add_argument(
    "--n_process",
    type=int,
    help="Number of processes to use",
    required=True,
)
parser.add_argument(
    "--expected_number_of_MSAs",
    type=int,
    help="Expected number of MSAs",
    required=True,
)
# parser.add_argument(
#     "--max_seqs",
#     type=int,
#     help="Maximum number of sequences to use per family",
#     required=True,
# )
# parser.add_argument(
#     "--max_sites",
#     type=int,
#     help="Maximum number of sites to use per family",
#     required=True,
# )
parser.add_argument(
    "--max_families",
    type=int,
    help="Maximum number of family to run on.",
    required=False,
    default=100000000
)


def name_internal_nodes(t: Tree):
    r"""
    Assigns names to the internal nodes of tree t if they don't already have a name.
    """
    def node_name_generator():
        """Generates unique node names for the tree."""
        internal_node_id = 1
        while True:
            yield f"internal-{internal_node_id}"
            internal_node_id += 1

    names = node_name_generator()

    def dfs_name_internal_nodes(p, v):
        global internal_node_id
        if v.name == '':
            v.name = next(names)
        if p:
            # print(f"{p.name} -> {v.name}")
            pass
        for u in v.get_children():
            dfs_name_internal_nodes(v, u)

    dfs_name_internal_nodes(None, t)


def create_node_name_vs_int_mappings(tree) -> Tuple[Dict, Dict]:
    node_name_to_int = {}
    int_to_node_name = {}

    def int_id_generator():
        """Generates unique node names for the tree."""
        key = 0
        while True:
            yield key
            key += 1

    int_id = int_id_generator()

    def dfs_create_mapping(v):
        key = next(int_id)
        node_name_to_int[v.name] = key
        int_to_node_name[key] = v.name
        for u in v.get_children():
            dfs_create_mapping(u)
    
    dfs_create_mapping(tree)

    return node_name_to_int, int_to_node_name

def write_out_tree(tree, node_name_to_int, tree_filepath) -> str:
    res = []

    def dfs_write_out_tree(p, v):
        if p:
            res.append(f"{node_name_to_int[p.name]} {node_name_to_int[v.name]}\n")
        for u in v.get_children():
            dfs_write_out_tree(v, u)

    dfs_write_out_tree(None, tree)

    with open(tree_filepath, "w") as file:
        file.write(f"{len(res)}\n")
        file.write(''.join(res))

def write_out_msa(msa: MSA, node_name_to_int, msa_filepath) -> str:
    res = ""
    nleaves = 0
    for (protein_name, sequence) in msa.get_msa().items():
        # Only write out sequences that are in the tree!
        if protein_name in node_name_to_int:
            res += f"{node_name_to_int[protein_name]} {sequence}\n"
            nleaves += 1
    with open(msa_filepath, "w") as outfile:
        outfile.write(f"{nleaves}\n")
        outfile.write(res)

def map_parsimony_indexing_back_to_str(int_to_node_name, cpp_parsimony_filepath, parsimony_filepath) -> None:
    # Read parsimony_filepath and replace each header by the node name, then write back to the same file.
    res = ""
    with open(cpp_parsimony_filepath, "r") as infile:
        with open(parsimony_filepath, "w") as outfile:
            for i, line in enumerate(infile):
                if i == 0:
                    res += line
                else:
                    line_contents = line.split(' ')
                    res += f"{int_to_node_name[int(line_contents[0])]} {line_contents[1]}"
            outfile.write(res)

def map_func(args):
    init_logger()
    logger = logging.getLogger("maximum_parsimony")
    a3m_dir = args[0]
    tree_dir = args[1]
    protein_family_name = args[2]
    outdir = args[3]
    max_seqs = args[4]
    max_sites = args[5]
    msa = MSA(
        a3m_dir=a3m_dir,
        protein_family_name=protein_family_name,
        max_seqs=max_seqs,
        max_sites=max_sites
    )
    # Read tree. Careful: Some trees might be corrupted / not exist due to MSA issues.
    try:
        tree = Tree(os.path.join(tree_dir, protein_family_name + ".newick"))
    except:
        logger.info(f"Malformed tree for family: {protein_family_name} . Skipping")
        return
    name_internal_nodes(tree)
    tree.write(format=3, outfile=os.path.join(outdir, protein_family_name + ".newick"))
    # Create input for C++ maximum parsimony
    node_name_to_int, int_to_node_name = create_node_name_vs_int_mappings(tree)

    # if True:
    #     if True:
    #         if True:
    #             tree_filepath = "cpp_tree.txt"
    #             msa_filepath = "cpp_msa.txt"
    #             cpp_parsimony_filepath = "cpp_parsimony.txt"

    with tempfile.NamedTemporaryFile("w") as tree_file:
        with tempfile.NamedTemporaryFile("w") as msa_file:
            with tempfile.NamedTemporaryFile("w") as parsimony_file:
                tree_filepath = tree_file.name
                msa_filepath = msa_file.name
                cpp_parsimony_filepath = parsimony_file.name

                # Write out C++ inputs
                write_out_tree(tree, node_name_to_int, tree_filepath)
                write_out_msa(msa, node_name_to_int, msa_filepath)

                # Run C++ maximum parsimony
                dir_path = os.path.dirname(os.path.realpath(__file__))
                os.system(f"{dir_path}/maximum_parsimony {tree_filepath} {msa_filepath} {cpp_parsimony_filepath}")
                # Convert .parsimony's indexing into string based.
                parsimony_filepath = os.path.join(outdir, protein_family_name + ".parsimony")
                map_parsimony_indexing_back_to_str(int_to_node_name, cpp_parsimony_filepath, parsimony_filepath)


if __name__ == "__main__":
    np.random.seed(1)

    # Pull out arguments
    args = parser.parse_args()
    a3m_dir = args.a3m_dir
    tree_dir = args.tree_dir
    n_process = args.n_process
    expected_number_of_MSAs = args.expected_number_of_MSAs
    outdir = args.outdir
    max_seqs = 0
    max_sites = 0
    max_families = args.max_families

    init_logger()
    logger = logging.getLogger("maximum_parsimony")
    logger.info("Starting ... ")

    if os.path.exists(outdir):
        raise ValueError(
            f"outdir {outdir} already exists. Aborting not to " f"overwrite!"
        )
    os.mkdir(outdir)

    if not os.path.exists(a3m_dir):
        raise ValueError(f"Could not find a3m_dir {a3m_dir}")

    filenames = list(os.listdir(a3m_dir))
    if not len(filenames) == expected_number_of_MSAs:
        raise ValueError(
            f"Number of MSAs is {len(filenames)}, does not match "
            f"expected {expected_number_of_MSAs}"
        )
    protein_family_names = [x.split(".")[0] for x in filenames][:max_families]

    map_args = [
        (a3m_dir, tree_dir, protein_family_name, outdir, max_seqs, max_sites)
        for protein_family_name in protein_family_names
    ]
    with multiprocessing.Pool(n_process) as pool:
        list(tqdm.tqdm(pool.imap(map_func, map_args), total=len(map_args)))
