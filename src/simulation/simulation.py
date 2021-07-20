r"""
Simulates MSA, contact maps, and ground truth ancestral states,
for end-to-end validation of our estimation pipeline.
"""
import multiprocessing
import os
import sys

import hashlib
import logging
import numpy as np
import pandas as pd
import random
import tqdm
from typing import Dict, List, Tuple

from ete3 import Tree

from src.maximum_parsimony import name_internal_nodes

sys.path.append("../")
import Phylo_util


def get_L(msa_path: str) -> int:
    with open(msa_path) as file:
        for i, line in enumerate(file):
            if i == 1:
                return len(line.strip())


def chain_jump(
    int_state: List[int],
    model_1: Phylo_util.substitution_model,
    model_2: Phylo_util.substitution_model,
    contacting_pairs: List[Tuple[int, int]],
    independent_sites: List[int],
    dist: float,
) -> List[int]:
    res = []
    for i in range(len(independent_sites)):
        res.append(Phylo_util.ending_state_without_expm(initial_state_index=int_state[i], model=model_1, t=dist))
    for i in range(len(contacting_pairs)):
        res.append(
            Phylo_util.ending_state_without_expm(initial_state_index=int_state[len(independent_sites) + i], model=model_2, t=dist)
        )
    return res


def chain_stationary(
    pi_1: np.array,
    pi_2: np.array,
    contacting_pairs: List[Tuple[int, int]],
    independent_sites: List[int],
) -> List[int]:
    res = []
    for _ in range(len(independent_sites)):
        res.append(np.random.choice(list(range(len(pi_1))), p=pi_1))
    for _ in range(len(contacting_pairs)):
        res.append(np.random.choice(list(range(len(pi_2))), p=pi_2))
    return res


def translate_states(
    int_states: Dict[str, List[int]],
    Q1_df: pd.DataFrame,
    Q2_df: pd.DataFrame,
    contacting_pairs: List[Tuple[int, int]],
    independent_sites: List[int],
) -> Dict[str, str]:
    res = {}  # type: Dict[str, str]
    L = len(contacting_pairs) + len(independent_sites)
    for v, int_state in int_states.items():
        state = ["@"] * L
        # First come the independent sites
        for i, site in enumerate(independent_sites):
            state[site] = Q1_df.index[int_state[i]]
        # Second come the co-evolving sites
        for i, (site_1, site_2) in enumerate(contacting_pairs):
            states = Q2_df.index[int_state[len(independent_sites) + i]]
            state_1, state_2 = states[0], states[1]
            state[site_1] = state_1
            state[site_2] = state_2
        assert all([s != "@" for s in state])
        res[v] = "".join(state)
    return res


def run_chain(
    tree: Tree,
    Q1_df: pd.DataFrame,
    Q2_df: pd.DataFrame,
    contacting_pairs: List[Tuple[int, int]],
    independent_sites: List[int],
) -> Dict[str, str]:
    r"""
    Run the chain down the tree, starting from the stationary distribution.
    Use Q1 for the independent sites, Q2 for the contacting sites.
    Returning a dictionary mapping node name to state.
    """
    Q1 = np.array(Q1_df)
    Q2 = np.array(Q2_df)
    pi_1 = Phylo_util.solve_stationery_dist(Q1)
    model_1 = Phylo_util.substitution_model(Q1, pi_1)
    pi_2 = Phylo_util.solve_stationery_dist(Q2)
    model_2 = Phylo_util.substitution_model(Q2, pi_2)
    int_states = {}  # type: Dict[str, List[int]]

    def dfs_run_chain(v):
        for u in v.get_children():
            # Set state of u
            int_states[u.name] = chain_jump(
                int_states[v.name], model_1, model_2, contacting_pairs, independent_sites, u.dist
            )
            dfs_run_chain(u)

    int_states[tree.name] = chain_stationary(pi_1, pi_2, contacting_pairs, independent_sites)
    dfs_run_chain(tree)

    states = translate_states(int_states, Q1_df, Q2_df, contacting_pairs, independent_sites)
    return states


def map_func(args: List) -> None:
    logger = logging.getLogger("phylo_correction.simulation")
    protein_family_name = args[0]
    a3m_dir = args[1]
    tree_dir = args[2]
    simulation_pct_interacting_positions = args[3]
    Q1_ground_truth = args[4]
    Q2_ground_truth = args[5]
    contact_simulated_dir = args[6]
    a3m_simulated_dir = args[7]
    ancestral_states_simulated_dir = args[8]
    use_cached = args[9]

    # Caching pattern: skip any computation as soon as possible
    contact_matrix_path = os.path.join(contact_simulated_dir, protein_family_name + ".cm")
    parsimony_tree_path = os.path.join(ancestral_states_simulated_dir, protein_family_name + ".newick")
    ancestral_states_path = os.path.join(ancestral_states_simulated_dir, protein_family_name + ".parsimony")
    output_msa_path = os.path.join(a3m_simulated_dir, protein_family_name + ".a3m")
    if use_cached and os.path.exists(contact_matrix_path) and os.path.exists(parsimony_tree_path) and os.path.exists(ancestral_states_path) and os.path.exists(output_msa_path):
        logger.info(f"Skipping. Cached simulation files for family {protein_family_name} at {contact_matrix_path} , {parsimony_tree_path} , {ancestral_states_path} , {output_msa_path}")
        return

    # Set seed for reproducibility
    seed = int(hashlib.md5((protein_family_name + "simulation").encode()).hexdigest()[:8], 16)
    logger.info(f"Starting on family {protein_family_name}. Setting random seed to: {seed}")
    np.random.seed(seed)
    random.seed(seed)

    # Read MSA
    msa_path = os.path.join(a3m_dir, protein_family_name + ".a3m")
    L = get_L(msa_path)
    logger.info(f"Processing family {protein_family_name}, L = {L}")

    # Read tree.
    try:
        tree = Tree(os.path.join(tree_dir, protein_family_name + ".newick"))
    except:
        logger.error(f"Malformed tree for family: {protein_family_name}")
        return
    name_internal_nodes(tree)
    # Write parsimony tree (is same as original tree but with internal nodes named)
    parsimony_tree_path = os.path.join(ancestral_states_simulated_dir, protein_family_name + ".newick")
    tree.write(format=3, outfile=os.path.join(parsimony_tree_path))
    os.system(f"chmod 555 {parsimony_tree_path}")

    # Read single-site transition rate matrix
    Q1 = pd.read_csv(Q1_ground_truth, sep="\t", index_col=0, keep_default_na=False, na_values=[""])
    # Read co-evolution rate matrix
    Q2 = pd.read_csv(Q2_ground_truth, sep="\t", index_col=0, keep_default_na=False, na_values=[""])
    states_list = list(Q1.index)
    # logger.info(f"Q1 states = {list(Q1.index)}")
    # logger.info(f"Q2 states = {list(Q2.index)}")
    # Pedantically check that states in Q1 and Q2 are compatible.
    if not (len(Q1.index) ** 2 == len(Q2.index)):
        logger.error("Q1 and Q2 indices not compatible")
        return
    for s1 in states_list:
        for s2 in states_list:
            if not (s1 + s2 in Q2.index):
                logger.error(f"Q1 and Q2 indices not compatible: {s1 + s2} not in Q2 index")
                return

    # Determine contacting pairs
    all_site_indices = list(range(L))
    np.random.shuffle(all_site_indices)
    contacting_pairs = [
        (all_site_indices[2 * i], all_site_indices[2 * i + 1])
        for i in range(int(L * simulation_pct_interacting_positions / 2.0))
    ]
    independent_sites = all_site_indices[len(contacting_pairs) :]
    contact_matrix = np.zeros(shape=(L, L), dtype=int)
    for (site1, site2) in contacting_pairs:
        contact_matrix[site1, site2] = contact_matrix[site2, site1] = 1
    for site in range(L):
        contact_matrix[site, site] = 1

    # Run Markov process down the tree
    states = run_chain(tree, Q1, Q2, contacting_pairs, independent_sites)

    # Write out contact matrix
    contact_matrix_path = os.path.join(contact_simulated_dir, protein_family_name + ".cm")
    np.savetxt(contact_matrix_path, contact_matrix, fmt="%d")
    os.system(f"chmod 555 {contact_matrix_path}")

    # Write out MSA
    msa = ""
    # First seq1 (the reference)
    msa += ">seq1\n"
    msa += states["seq1"] + "\n"
    # Now all other states
    for leaf in tree:
        if leaf.name != "seq1":
            msa += ">" + leaf.name + "\n"
            msa += states[leaf.name] + "\n"
    output_msa_path = os.path.join(a3m_simulated_dir, protein_family_name + ".a3m")
    with open(output_msa_path, "w") as file:
        file.write(msa)
    os.system(f"chmod 555 {output_msa_path}")

    # Write out ancestral states
    num_nodes = 0
    maximum_parsimony = ""
    for node in tree.traverse("levelorder"):
        num_nodes += 1
        maximum_parsimony += node.name + " " + states[node.name] + "\n"
    ancestral_states_path = os.path.join(ancestral_states_simulated_dir, protein_family_name + ".parsimony")
    with open(ancestral_states_path, "w") as file:
        file.write(f"{num_nodes}\n" + maximum_parsimony)
    os.system(f"chmod 555 {ancestral_states_path}")


class Simulator:
    r"""
    Simulate MSAs and contact maps given a tree.

    Args:
        a3m_dir: Directory where the original MSA files (.a3m) are found.
            They are only used to determine the length (number of amino acids)
            of each protein family.
        tree_dir: Directory where the tree files (.newick) are found.
        a3m_simulated_dir: Directory where to write the simulated MSAs files (.a3m).
        contact_simulated_dir: Directory where to write the simulated contact map files (.cm).
        ancestral_states_simulated_dir: The ancestral states of the simulation will be
            stored here. These files look just like for the MaximumParsimonyReconstructor,
            but contain ground truth ancestral states instead of estimated states.
            (This is the whole purpose of simulation! To see how maximum parsimony
            screws up our estimates.)
        n_process: How many processes to use to parallelize computation.
        expected_number_of_MSAs: The number of files in a3m_dir. This argument
            is only used to sanity check that the correct a3m_dir is being used.
            It has no functional implications.
        max_families: Only simulate MSAs for 'max_families' randomly chosen files in a3m_dir.
            This is useful for testing and to see what happens if less data is used.
        simulation_pct_interacting_positions: What percent of the positions in each protein
            family will be in contact.
        Q1_ground_truth: Rate matrix for the evolution of single sites.
        Q2_ground_truth: Rate matrix for pairs of sites that are in contact.
        use_cached: If True and an output file already exists, all computation will be skipped.
    """
    def __init__(
        self,
        a3m_dir: str,
        tree_dir: str,
        a3m_simulated_dir: str,
        contact_simulated_dir: str,
        ancestral_states_simulated_dir: str,
        n_process: int,
        expected_number_of_MSAs: int,
        max_families: int,
        simulation_pct_interacting_positions,
        Q1_ground_truth: str,
        Q2_ground_truth: str,
        use_cached: bool = False,
    ):
        self.a3m_dir = a3m_dir
        self.tree_dir = tree_dir
        self.a3m_simulated_dir = a3m_simulated_dir
        self.contact_simulated_dir = contact_simulated_dir
        self.ancestral_states_simulated_dir = ancestral_states_simulated_dir
        self.n_process = n_process
        self.expected_number_of_MSAs = expected_number_of_MSAs
        self.max_families = max_families
        self.simulation_pct_interacting_positions = simulation_pct_interacting_positions
        self.Q1_ground_truth = Q1_ground_truth
        self.Q2_ground_truth = Q2_ground_truth
        self.use_cached = use_cached

    def run(self) -> None:
        a3m_dir = self.a3m_dir
        tree_dir = self.tree_dir
        a3m_simulated_dir = self.a3m_simulated_dir
        contact_simulated_dir = self.contact_simulated_dir
        ancestral_states_simulated_dir = self.ancestral_states_simulated_dir
        n_process = self.n_process
        expected_number_of_MSAs = self.expected_number_of_MSAs
        max_families = self.max_families
        simulation_pct_interacting_positions = self.simulation_pct_interacting_positions
        Q1_ground_truth = self.Q1_ground_truth
        Q2_ground_truth = self.Q2_ground_truth
        use_cached = self.use_cached

        logger = logging.getLogger("phylo_correction.simulation")
        logger.info("Starting ... ")

        for dire in [a3m_dir, tree_dir]:
            if not os.path.exists(dire):
                raise ValueError(f"Could not find directory {dire}")

        for dire in [a3m_simulated_dir, contact_simulated_dir, ancestral_states_simulated_dir]:
            if os.path.exists(dire) and not use_cached:
                raise ValueError(f"outdir {dire} already exists. Aborting not to " f"overwrite!")
            if not os.path.exists(dire):
                os.makedirs(dire)

        filenames = sorted(list(os.listdir(a3m_dir)))
        random.Random(123).shuffle(filenames)
        if not len(filenames) == expected_number_of_MSAs:
            raise ValueError(
                f"Number of MSAs is {len(filenames)}, does not match " f"expected {expected_number_of_MSAs}"
            )
        protein_family_names = [x.split(".")[0] for x in filenames][:max_families]

        map_args = [
            [
                protein_family_name,
                a3m_dir,
                tree_dir,
                simulation_pct_interacting_positions,
                Q1_ground_truth,
                Q2_ground_truth,
                contact_simulated_dir,
                a3m_simulated_dir,
                ancestral_states_simulated_dir,
                use_cached,
            ]
            for protein_family_name in protein_family_names
        ]
        if n_process > 1:
            with multiprocessing.Pool(n_process) as pool:
                list(tqdm.tqdm(pool.imap(map_func, map_args), total=len(map_args)))
        else:
            list(tqdm.tqdm(map(map_func, map_args), total=len(map_args)))
