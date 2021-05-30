r"""
Simulates MSA, contact maps, and ground truth ancestral states,
for end-to-end validation of our estimation pipeline.
"""
import argparse
import multiprocessing
import os
import sys

import hashlib
import logging
import numpy as np
import pandas as pd
import random
import tempfile
import tqdm
from typing import Dict, List, Tuple

from ete3 import Tree

sys.path.append('../')
from maximum_parsimony import name_internal_nodes
import Phylo_util


def init_logger():
    logger = logging.getLogger("simulation")
    logger.setLevel(logging.DEBUG)
    # fmt_str = "[%(asctime)s] %(levelname)s @ line %(lineno)d: %(message)s"
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler("simulation.log")
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)


parser = argparse.ArgumentParser(description="Data simulation.")
parser.add_argument(
    "--a3m_dir",
    type=str,
    help="Directory where the ground truth MSAs are found (.a3m files). This is just used to figure out the number of amino acids of each protein family.",
    required=True,
)
parser.add_argument(
    "--tree_dir",
    type=str,
    help="Directory where the ground truth trees are found (.newick files)",
    required=True,
)
parser.add_argument(
    "--a3m_simulated_dir",
    type=str,
    help="Directory where the simulated MSAs will be found.",
    required=True,
)
parser.add_argument(
    "--contact_simulated_dir",
    type=str,
    help="Directory where the simulated contact maps will be found.",
    required=True,
)
parser.add_argument(
    "--ancestral_states_simulated_dir",
    type=str,
    help="Directory where the simulated ancestral states will be found",
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
parser.add_argument(
    "--max_families",
    type=int,
    help="Maximum number of family to run on.",
    required=False,
    default=100000000
)
parser.add_argument(
    "--simulation_pct_interacting_positions",
    type=float,
    help="Pct of sites that will be interacting",
    required=True,
)
parser.add_argument(
    "--Q1_ground_truth",
    type=str,
    help="Directory where the ground truth single-site transition rate matrix lies",
    required=True,
)
parser.add_argument(
    "--Q2_ground_truth",
    type=str,
    help="Directory where the ground truth co-evolution transition rate matrix lies",
    required=True,
)


def get_L(msa_path: str) -> int:
    with open(msa_path) as file:
        for i, line in enumerate(file):
            if i == 1:
                return len(line.strip())


def chain_jump(int_state: List[int], model_1, model_2, contacting_pairs, independent_sites, dist) -> List[int]:
    res = []
    for i in range(len(independent_sites)):
        res.append(Phylo_util.ending_state(initial_state_index=int_state[i], model=model_1, t=dist))
    for i in range(len(contacting_pairs)):
        res.append(Phylo_util.ending_state(initial_state_index=int_state[len(independent_sites) + i], model=model_2, t=dist))
    return res


def chain_stationary(pi_1, pi_2, contacting_pairs, independent_sites) -> List[int]:
    res = []
    for _ in range(len(independent_sites)):
        res.append(
            np.random.choice(
                list(range(len(pi_1))),
                p=pi_1
            )
        )
    for _ in range(len(contacting_pairs)):
        res.append(
            np.random.choice(
                list(range(len(pi_2))),
                p=pi_2
            )
        )
    return res


def translate_states(int_states, Q1_df, Q2_df, contacting_pairs, independent_sites) -> Dict:
    res = {}
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
        assert(all([s != "@" for s in state]))
        res[v] = "".join(state)
    return res


def run_chain(tree, Q1_df, Q2_df, contacting_pairs, independent_sites) -> Dict:
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
    int_states = {}

    def dfs_run_chain(v):
        for u in v.get_children():
            # Set state of u
            int_states[u.name] = chain_jump(int_states[v.name], model_1, model_2, contacting_pairs, independent_sites, u.dist)
            dfs_run_chain(u)

    int_states[tree.name] = chain_stationary(pi_1, pi_2, contacting_pairs, independent_sites)
    dfs_run_chain(tree)

    states = translate_states(int_states, Q1_df, Q2_df, contacting_pairs, independent_sites)
    return states


def map_func(args):
    logger = logging.getLogger("simulation")
    protein_family_name = args[0]
    a3m_dir = args[1]
    tree_dir = args[2]
    simulation_pct_interacting_positions = args[3]
    Q1_ground_truth = args[4]
    Q2_ground_truth = args[5]
    contact_simulated_dir = args[6]
    a3m_simulated_dir = args[7]
    ancestral_states_simulated_dir = args[8]

    # Set seed for reproducibility
    seed = int(hashlib.md5((protein_family_name + 'simulation').encode()).hexdigest()[:8], 16)
    logger.info(f"Setting random seed to: {seed}")
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
    parsimony_tree_path = os.path.join(ancestral_states_simulated_dir, protein_family_name + '.newick')
    tree.write(format=3, outfile=os.path.join(parsimony_tree_path))

    # Read single-site transition rate matrix
    Q1 = pd.read_csv(Q1_ground_truth, sep="\t", index_col=0, keep_default_na=False, na_values=[''])
    # Read co-evolution rate matrix
    Q2 = pd.read_csv(Q2_ground_truth, sep="\t", index_col=0, keep_default_na=False, na_values=[''])
    states = list(Q1.index)
    # logger.info(f"Q1 states = {list(Q1.index)}")
    # logger.info(f"Q2 states = {list(Q2.index)}")
    # Pedantically check that states in Q1 and Q2 are compatible.
    if not (len(Q1.index) ** 2 == len(Q2.index)):
        logger.error("Q1 and Q2 indices not compatible")
        return
    for s1 in states:
        for s2 in states:
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
    independent_sites = all_site_indices[len(contacting_pairs):]
    contact_matrix = np.zeros(shape=(L, L), dtype=int)
    for (site1, site2) in contacting_pairs:
        contact_matrix[site1, site2] = contact_matrix[site2, site1] = 1
    for site in range(L):
        contact_matrix[site, site] = 1

    # Run Markov process down the tree
    states = run_chain(tree, Q1, Q2, contacting_pairs, independent_sites)

    # Write out contact matrix
    contact_matrix_path = os.path.join(contact_simulated_dir, protein_family_name + '.cm')
    np.savetxt(contact_matrix_path, contact_matrix, fmt="%d")

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
    msa_path = os.path.join(a3m_simulated_dir, protein_family_name + '.a3m')
    with open(msa_path, "w") as file:
        file.write(msa)

    # Write out ancestral states
    num_nodes = 0
    maximum_parsimony = ""
    for node in tree.traverse("levelorder"):
        num_nodes += 1
        maximum_parsimony += node.name + " " + states[node.name] + "\n"
    ancestral_states_path = os.path.join(ancestral_states_simulated_dir, protein_family_name + '.parsimony')
    with open(ancestral_states_path, "w") as file:
        file.write(f"{num_nodes}\n" + maximum_parsimony)


class Simulator:
    def __init__(
        self,
        a3m_dir,
        tree_dir,
        a3m_simulated_dir,
        contact_simulated_dir,
        ancestral_states_simulated_dir,
        n_process,
        expected_number_of_MSAs,
        max_families,
        simulation_pct_interacting_positions,
        Q1_ground_truth,
        Q2_ground_truth,
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

    def run(self):
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

        init_logger()
        logger = logging.getLogger("simulation")
        logger.info("Starting ... ")

        for dire in [a3m_dir, tree_dir]:
            if not os.path.exists(dire):
                raise ValueError(f"Could not find directory {dire}")

        for dire in [a3m_simulated_dir, contact_simulated_dir, ancestral_states_simulated_dir]:
            if os.path.exists(dire):
                raise ValueError(
                    f"outdir {dire} already exists. Aborting not to " f"overwrite!"
                )
            os.makedirs(dire)

        filenames = list(os.listdir(a3m_dir))
        if not len(filenames) == expected_number_of_MSAs:
            raise ValueError(
                f"Number of MSAs is {len(filenames)}, does not match "
                f"expected {expected_number_of_MSAs}"
            )
        protein_family_names = [x.split(".")[0] for x in filenames][:max_families]

        map_args = [
            (protein_family_name,
             a3m_dir,
             tree_dir,
             simulation_pct_interacting_positions,
             Q1_ground_truth,
             Q2_ground_truth,
             contact_simulated_dir,
             a3m_simulated_dir,
             ancestral_states_simulated_dir)
            for protein_family_name in protein_family_names
        ]
        if n_process > 1:
            with multiprocessing.Pool(n_process) as pool:
                list(tqdm.tqdm(pool.imap(map_func, map_args), total=len(map_args)))
        else:
            list(tqdm.tqdm(map(map_func, map_args), total=len(map_args)))


def _main():
    # Pull out arguments
    args = parser.parse_args()
    simulator = Simulator(
        a3m_dir=args.a3m_dir,
        tree_dir=args.tree_dir,
        a3m_simulated_dir=args.a3m_simulated_dir,
        contact_simulated_dir=args.contact_simulated_dir,
        ancestral_states_simulated_dir=args.ancestral_states_simulated_dir,
        n_process=args.n_process,
        expected_number_of_MSAs=args.expected_number_of_MSAs,
        max_families=args.max_families,
        simulation_pct_interacting_positions=args.simulation_pct_interacting_positions,
        Q1_ground_truth=args.Q1_ground_truth,
        Q2_ground_truth=args.Q2_ground_truth,
    )
    simulator.run()


if __name__ == "__main__":
    _main()
