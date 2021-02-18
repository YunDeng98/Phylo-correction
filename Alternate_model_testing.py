import numpy as np
import scipy as sp
import pandas as pd
import random
import ete3
from ete3 import Tree, TreeNode
import Phylo_simulator
import Felsenstein_algorithm
import ngesh
from timeit import default_timer as timer

amino_acid = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


def rate_matrix_unit_modifier(new_rate_matrix, state, Z):
    n = new_rate_matrix.shape[0]
    for i in range(n):
        if n != state:
            new_rate_matrix[state, i] = Z * new_rate_matrix[state, i]
            new_rate_matrix[i, state] = Z * new_rate_matrix[i, state]


def rate_matrix_modifier(rate_matrix, favorable_states, Z):
    n = len(favorable_states)
    new_rate_matrix = np.copy(rate_matrix)
    for i in range(n):
        rate_matrix_unit_modifier(new_rate_matrix, favorable_states[i], Z)
    for i in range(n):
        new_rate_matrix[i, i] = -sum(new_rate_matrix[i,]) + new_rate_matrix[i, i]
    return new_rate_matrix


def solve_stationery_dist(rate_matrix):
    eigvals, eigvecs = np.linalg.eig(rate_matrix.transpose())
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    eigvals = np.abs(eigvals)
    index = np.argmin(eigvals)
    stationery_dist = eigvecs[:, index]
    stationery_dist = stationery_dist / sum(stationery_dist)
    return stationery_dist


def build_sequence_alignment(tree, rate_matrix, n):
    seq_dict = Phylo_simulator.build_empty_seq_dict(tree)
    stationery_dist = solve_stationery_dist(rate_matrix)
    all_states = [i for i in range(len(stationery_dist))]
    for i in range(n):
        root_state = random.choices(population=all_states, weights=stationery_dist, k=1)
        Phylo_simulator.add_variation(tree, root_state[0], rate_matrix, seq_dict)
    return seq_dict


def permute_model(rate_matrix, stationery_dist):
    n = rate_matrix.shape[0]
    mid_rate_matrix = np.zeros((n, n))
    new_rate_matrix = np.zeros((n, n))
    new_stationery_dist = np.zeros(n)
    permute_order = np.random.permutation(n)
    for i in range(n):
        mid_rate_matrix[:, i] = rate_matrix[:, permute_order[i]]
    for i in range(n):
        new_rate_matrix[i, :] = mid_rate_matrix[permute_order[i], :]
    for i in range(n):
        new_stationery_dist[i] = stationery_dist[permute_order[i]]
    return new_rate_matrix, new_stationery_dist


rate_matrix = np.loadtxt('WAG_matrix.txt')
new_rate_matrix = np.loadtxt('WAG_new_matrix.txt')
stationery_dist = np.loadtxt('WAG_stationery_dist.txt')
product_matrix = Phylo_simulator.matrix_product(rate_matrix)
new_product_matrix = Phylo_simulator.matrix_product(new_rate_matrix)
product_stationery_dist = np.zeros(400)
for i in range(20):
    for j in range(20):
        product_stationery_dist[(i - 1)*20 + j] = stationery_dist[i]*stationery_dist[j]
new_product_stationery_dist = [0.0025 for i in range(400)]
tree = Tree("test_tree.nw", format=1)
record = np.zeros((50, 3))
for i in range(50):
    print(i)
    seq_dict = build_sequence_alignment(tree, new_product_matrix, 1)
    l1 = Felsenstein_algorithm.tree_log_likelihood(tree, seq_dict, 0, product_matrix, product_stationery_dist)
    l2 = Felsenstein_algorithm.tree_log_likelihood(tree, seq_dict, 0, new_product_matrix, new_product_stationery_dist)
    print(l1, l2, l1 - l2)
    record[i, 0] = l1
    record[i, 1] = l2
    record[i, 2] = l1 - l2
np.savetxt('reverse_record.txt', record)