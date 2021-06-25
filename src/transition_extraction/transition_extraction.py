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
    tree: Tree,
    sequences: Dict[str, str],
) -> List[Tuple[str, str, float, float, int, str, str, int]]:
    # The root's name was not written out by ete3 in the maximum_parsimony script,
    # so we name it ourselves.
    assert tree.name == ""
    tree.name = "internal-1"
    res = []  # type: List[Tuple[str, str, float, float, int, str, str, int]]
    height = {}  # type: Dict[str, float]
    path_height = {}  # type: Dict[str, int]

    def dfs_get_transitions(v, site_id):
        height[v.name] = 0
        path_height[v.name] = 0
        for u in v.get_children():
            dfs_get_transitions(u, site_id)
            height[v.name] = max(height[v.name], height[u.name] + u.dist)
            path_height[v.name] = max(path_height[v.name], path_height[u.name] + 1)
        for u in v.get_children():
            res.append(
                (
                    sequences[v.name][site_id],
                    sequences[u.name][site_id],
                    u.dist,
                    height[v.name],
                    path_height[v.name],
                    v.name,
                    u.name,
                    site_id,
                )
            )

    L = len(sequences["internal-1"])
    for site_id in range(L):
        dfs_get_transitions(tree, site_id)
    return res


def map_func(args: List) -> None:
    # a3m_dir = args[0]
    parsimony_dir = args[1]
    protein_family_name = args[2]
    outdir = args[3]
    use_cached = args[4]

    logger = logging.getLogger("transition_extraction")

    # Caching pattern: skip any computation as soon as possible
    transition_filename = os.path.join(outdir, protein_family_name + ".transitions")
    if use_cached and os.path.exists(transition_filename):
        logger.info(f"Skipping. Cached transitions for family {protein_family_name} at {transition_filename}")
        return

    seed = int(hashlib.md5((protein_family_name + "transition_extraction").encode()).hexdigest()[:8], 16)
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
    sequences = {}
    with open(os.path.join(parsimony_dir, protein_family_name + ".parsimony"), "r") as infile:
        for i, line in enumerate(infile):
            line_contents = line.split(" ")
            if i != 0:
                sequences[line_contents[0]] = line_contents[1].rstrip("\n")

    transitions = get_transitions(tree, sequences)
    res = "starting_state,ending_state,length,height,path_height,starting_node,ending_node,site_id\n"
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
            + "\n"
        )

    transition_filename = os.path.join(outdir, protein_family_name + ".transitions")
    with open(transition_filename, "w") as transition_file:
        transition_file.write(res)


class TransitionExtractor:
    def __init__(
        self,
        a3m_dir: str,
        parsimony_dir: str,
        n_process: int,
        expected_number_of_MSAs: int,
        outdir: str,
        max_families: int,
        use_cached: bool = False,
    ):
        self.a3m_dir = a3m_dir
        self.parsimony_dir = parsimony_dir
        self.n_process = n_process
        self.expected_number_of_MSAs = expected_number_of_MSAs
        self.outdir = outdir
        self.max_families = max_families
        self.use_cached = use_cached

    def run(self) -> None:
        a3m_dir = self.a3m_dir
        parsimony_dir = self.parsimony_dir
        n_process = self.n_process
        expected_number_of_MSAs = self.expected_number_of_MSAs
        outdir = self.outdir
        max_families = self.max_families
        use_cached = self.use_cached

        logger = logging.getLogger("transition_extraction")
        logger.info("Starting ... ")

        if os.path.exists(outdir) and not use_cached:
            raise ValueError(f"outdir {outdir} already exists. Aborting not to " f"overwrite!")

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if not os.path.exists(a3m_dir):
            raise ValueError(f"Could not find a3m_dir {a3m_dir}")

        filenames = sorted(list(os.listdir(a3m_dir)))
        if not len(filenames) == expected_number_of_MSAs:
            raise ValueError(
                f"Number of MSAs is {len(filenames)}, does not match " f"expected {expected_number_of_MSAs}"
            )
        protein_family_names = [x.split(".")[0] for x in filenames][:max_families]

        # print(f"protein_family_names = {protein_family_names}")

        map_args = [
            [a3m_dir, parsimony_dir, protein_family_name, outdir, use_cached]
            for protein_family_name in protein_family_names
        ]
        if n_process > 1:
            with multiprocessing.Pool(n_process) as pool:
                list(tqdm.tqdm(pool.imap(map_func, map_args), total=len(map_args)))
        else:
            list(tqdm.tqdm(map(map_func, map_args), total=len(map_args)))

        os.system(f"chmod -R 555 {outdir}")
