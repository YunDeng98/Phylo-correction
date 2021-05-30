r"""
Runs maximum parsimony reconstruction for a given protein family.
Uses call to C++ implementation.
"""
import multiprocessing
import os
import sys

import logging
import numpy as np
import tempfile
import tqdm
from typing import Dict, Optional, Tuple
import hashlib
import random

from ete3 import Tree

from src.phylogeny_generation import MSA

sys.path.append("../")


def name_internal_nodes(t: Tree) -> None:
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

    def dfs_name_internal_nodes(p: Optional[Tree], v: Tree) -> None:
        global internal_node_id
        if v.name == "":
            v.name = next(names)
        if p:
            # print(f"{p.name} -> {v.name}")
            pass
        for u in v.get_children():
            dfs_name_internal_nodes(v, u)

    dfs_name_internal_nodes(None, t)


def create_node_name_vs_int_mappings(
    tree: Tree,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    node_name_to_int = {}  # type: Dict[str, int]
    int_to_node_name = {}  # type: Dict[int, str]

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


def write_out_tree(
    tree: Tree,
    node_name_to_int: Dict[str, int],
    tree_filepath: str,
) -> None:
    res = []

    def dfs_write_out_tree(p, v):
        if p:
            res.append(f"{node_name_to_int[p.name]} {node_name_to_int[v.name]}\n")
        for u in v.get_children():
            dfs_write_out_tree(v, u)

    dfs_write_out_tree(None, tree)

    with open(tree_filepath, "w") as file:
        file.write(f"{len(res) + 1}\n")
        file.write("".join(res))


def write_out_msa(
    msa: MSA,
    node_name_to_int: Dict[str, int],
    msa_filepath: str,
) -> None:
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


def map_parsimony_indexing_back_to_str(
    int_to_node_name: Dict[int, str],
    cpp_parsimony_filepath: str,
    parsimony_filepath: str,
) -> None:
    # Read parsimony_filepath and replace each header by the node name, then write back to the same file.
    res = ""
    with open(cpp_parsimony_filepath, "r") as infile:
        with open(parsimony_filepath, "w") as outfile:
            for i, line in enumerate(infile):
                if i == 0:
                    res += line
                else:
                    line_contents = line.split(" ")
                    res += f"{int_to_node_name[int(line_contents[0])]} {line_contents[1]}"
            outfile.write(res)


def map_func(args) -> None:
    a3m_dir = args[0]
    tree_dir = args[1]
    protein_family_name = args[2]
    outdir = args[3]
    max_seqs = args[4]
    max_sites = args[5]

    logger = logging.getLogger("maximum_parsimony")
    seed = int(hashlib.md5((protein_family_name + "maximum_parsimony").encode()).hexdigest()[:8], 16)
    logger.info(f"Setting random seed to: {seed}")
    np.random.seed(seed)
    random.seed(seed)

    msa = MSA(a3m_dir=a3m_dir, protein_family_name=protein_family_name, max_seqs=max_seqs, max_sites=max_sites)
    # Read tree. Careful: Some trees might be corrupted / not exist due to MSA issues.
    try:
        tree = Tree(os.path.join(tree_dir, protein_family_name + ".newick"))
    except:
        logger.error(f"Malformed tree for family: {protein_family_name}")
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
                call_result = os.system(
                    f"{dir_path}/maximum_parsimony {tree_filepath} {msa_filepath} {cpp_parsimony_filepath}"
                )
                # logger.info(f"Call result for family {protein_family_name} = {call_result}")
                if call_result != 0:
                    logger.error(f"Failed to run C++ maximum parsimony on family {protein_family_name}")
                # Convert .parsimony's indexing into string based.
                try:
                    parsimony_filepath = os.path.join(outdir, protein_family_name + ".parsimony")
                    map_parsimony_indexing_back_to_str(int_to_node_name, cpp_parsimony_filepath, parsimony_filepath)
                except:
                    logger.error(f"Failed to process C++ output for {protein_family_name}")


class MaximumParsimonyReconstructor:
    def __init__(
        self,
        a3m_dir: str,
        tree_dir: str,
        n_process: int,
        expected_number_of_MSAs: int,
        outdir: str,
        max_families: int,
    ):
        logger = logging.getLogger("maximum_parsimony")
        self.a3m_dir = a3m_dir
        self.tree_dir = tree_dir
        self.n_process = n_process
        self.expected_number_of_MSAs = expected_number_of_MSAs
        self.outdir = outdir
        self.max_families = max_families
        dir_path = os.path.dirname(os.path.realpath(__file__))
        maximum_parsimony_bin_path = os.path.join(dir_path, "maximum_parsimony")
        maximum_parsimony_path = os.path.join(dir_path, "maximum_parsimony.cpp")
        if not os.path.exists(maximum_parsimony_bin_path):
            logger.info("Compiling maximum_parsimony.cpp")
            os.system(
                "g++ -std=c++17 -O3 -Wshadow -Wall  -Wextra -D_GLIBCXX_DEBUG"
                f" -o {maximum_parsimony_bin_path} {maximum_parsimony_path}"
            )
            # Test maximum_parsimony.cpp with:
            # $ ./maximum_parsimony test_data/tree.txt test_data/sequences.txt
            #   test_data/solution.txt

    def run(self) -> None:
        a3m_dir = self.a3m_dir
        tree_dir = self.tree_dir
        n_process = self.n_process
        expected_number_of_MSAs = self.expected_number_of_MSAs
        outdir = self.outdir
        max_families = self.max_families

        max_seqs = 0
        max_sites = 0

        logger = logging.getLogger("maximum_parsimony")
        logger.info("Starting ... ")

        if os.path.exists(outdir):
            raise ValueError(f"outdir {outdir} already exists. Aborting not to " f"overwrite!")
        os.makedirs(outdir)

        if not os.path.exists(a3m_dir):
            raise ValueError(f"Could not find a3m_dir {a3m_dir}")

        filenames = list(os.listdir(a3m_dir))
        if not len(filenames) == expected_number_of_MSAs:
            raise ValueError(
                f"Number of MSAs is {len(filenames)}, does not match " f"expected {expected_number_of_MSAs}"
            )
        protein_family_names = [x.split(".")[0] for x in filenames][:max_families]

        map_args = [
            [a3m_dir, tree_dir, protein_family_name, outdir, max_seqs, max_sites]
            for protein_family_name in protein_family_names
        ]
        if n_process > 1:
            with multiprocessing.Pool(n_process) as pool:
                list(tqdm.tqdm(pool.imap(map_func, map_args), total=len(map_args)))
        else:
            list(tqdm.tqdm(map(map_func, map_args), total=len(map_args)))
