r"""
Reads phylogenies with ancestral states and produces a list of all transitions observed,
annotated with useful information such as height, branch length, etc.
"""
import multiprocessing
import os
import sys

import logging
import numpy as np
import tqdm
import random
import hashlib

from typing import Dict, List, Tuple

from ete3 import Tree

sys.path.append("../")


def get_transitions(
    tree: Tree, sequences: Dict[str, str], contact_matrix: np.array
) -> List[Tuple[str, str, float, float, int, str, str, int, int]]:
    # The root's name was not written out by ete3 in the maximum_parsimony script,
    # so we name it ourselves.
    assert tree.name == ""
    tree.name = "internal-1"
    res = []  # type: List[Tuple[str, str, float, float, int, str, str, int, int]]
    height = {}  # type: Dict[str, float]
    path_height = {}  # type: Dict[str, int]

    def dfs_get_transitions(v, site1_id, site2_id):
        height[v.name] = 0
        path_height[v.name] = 0
        for u in v.get_children():
            dfs_get_transitions(u, site1_id, site2_id)
            height[v.name] = max(height[v.name], height[u.name] + u.dist)
            path_height[v.name] = max(path_height[v.name], path_height[u.name] + 1)
        for u in v.get_children():
            res.append(
                (
                    sequences[v.name][site1_id] + sequences[v.name][site2_id],
                    sequences[u.name][site1_id] + sequences[u.name][site2_id],
                    u.dist,
                    height[v.name],
                    path_height[v.name],
                    v.name,
                    u.name,
                    site1_id,
                    site2_id,
                )
            )

    L = len(sequences["internal-1"])
    assert contact_matrix.shape == (L, L)
    sites_in_contact = np.array(
        [(i, j) for i in range(L) for j in range(i + 1, L) if contact_matrix[i, j] == 1 and abs(i - j) >= 7]
    )
    # chosen_sites_in_contact = sites_in_contact[
    #     np.random.choice(
    #         range(len(sites_in_contact)),
    #         size=min(L, len(sites_in_contact)),
    #         replace=False)]
    chosen_sites_in_contact = sites_in_contact  # Just choose all
    for (site1_id, site2_id) in chosen_sites_in_contact:
        assert contact_matrix[site1_id, site2_id] == 1
        assert contact_matrix[site2_id, site1_id] == 1
        dfs_get_transitions(tree, site1_id, site2_id)
    return res


def map_func(args: List) -> None:
    # a3m_dir = args[0]
    parsimony_dir = args[1]
    protein_family_name = args[2]
    outdir = args[3]
    contact_dir = args[4]

    logger = logging.getLogger("co_transition_extraction")
    seed = int(hashlib.md5((protein_family_name + "co_transition_extraction").encode()).hexdigest()[:8], 16)
    logger.info(f"Setting random seed to: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"Starting on family {protein_family_name}")

    # Read tree. Careful: Some trees might be corrupted / not exist due to MSA issues.
    try:
        tree = Tree(os.path.join(parsimony_dir, protein_family_name + ".newick"), format=3)
    except:
        logger.error(f"Malformed tree for family: {protein_family_name}")
        return
    # Read sequences
    sequences = {}  # type: Dict[str, str]
    with open(os.path.join(parsimony_dir, protein_family_name + ".parsimony"), "r") as infile:
        for i, line in enumerate(infile):
            line_contents = line.split(" ")
            if i != 0:
                sequences[line_contents[0]] = line_contents[1].rstrip("\n")

    # Read contact matrix
    contact_matrix = np.loadtxt(os.path.join(contact_dir, protein_family_name + ".cm"))

    transitions = get_transitions(tree, sequences, contact_matrix)
    res = "starting_state,ending_state,length,height,path_height,starting_node,ending_node,site1_id,site2_id\n"
    for transition in transitions:
        res += (
            transition[0]
            + ","
            + transition[1]
            + ","
            + str(transition[2])
            + ","
            + str(transition[3])
            + ","
            + str(transition[4])
            + ","
            + str(transition[5])
            + ","
            + str(transition[6])
            + ","
            + str(transition[7])
            + ","
            + str(transition[8])
            + "\n"
        )

    transition_filename = os.path.join(outdir, protein_family_name + ".transitions")
    with open(transition_filename, "w") as transition_file:
        transition_file.write(res)


class CoTransitionExtractor:
    def __init__(
        self,
        a3m_dir: str,
        parsimony_dir: str,
        n_process: int,
        expected_number_of_MSAs: int,
        outdir: str,
        max_families: int,
        contact_dir: str,
    ):
        self.a3m_dir = a3m_dir
        self.parsimony_dir = parsimony_dir
        self.n_process = n_process
        self.expected_number_of_MSAs = expected_number_of_MSAs
        self.outdir = outdir
        self.max_families = max_families
        self.contact_dir = contact_dir

    def run(self) -> None:
        a3m_dir = self.a3m_dir
        parsimony_dir = self.parsimony_dir
        n_process = self.n_process
        expected_number_of_MSAs = self.expected_number_of_MSAs
        outdir = self.outdir
        max_families = self.max_families
        contact_dir = self.contact_dir

        logger = logging.getLogger("co_transition_extraction")
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

        # print(f"protein_family_names = {protein_family_names}")

        map_args = [
            (a3m_dir, parsimony_dir, protein_family_name, outdir, contact_dir)
            for protein_family_name in protein_family_names
        ]
        if n_process > 1:
            with multiprocessing.Pool(n_process) as pool:
                list(tqdm.tqdm(pool.imap(map_func, map_args), total=len(map_args)))
        else:
            list(tqdm.tqdm(map(map_func, map_args), total=len(map_args)))
