import numpy as np
import scipy as sp
import random
import ete3
from ete3 import Tree, TreeNode

# a function which can determine the waiting time given a current state and its next state.
# Input: a current state index and the rate matrix. Output: the waiting time and the next state.
from numpy import ndarray


def transition_info(current_state, rate_matrix):
    num_states = rate_matrix.shape[0]
    leaving_rate = rate_matrix[current_state, current_state]
    waiting_time = np.random.exponential(-1 / leaving_rate)
    rate_list = rate_matrix[current_state, :]
    rate_list = np.maximum(rate_list, 0)
    all_states = [i for i in range(num_states)]
    next_state = random.choices(population=all_states, weights=rate_list, k=1)
    return waiting_time, next_state[0]


# a function which can determine the ending state after a period of time given the initial state.
# Input: an initial state index, rate matrix, and the length of the period of time. Output: the ending state.
def ending_state(initial_state_index, rate_matrix, t):
    current_state_index = initial_state_index
    remaining_time = t
    while remaining_time > 0:
        waiting_time, next_state_index = transition_info(current_state_index, rate_matrix)
        remaining_time -= waiting_time
        current_state_index = next_state_index
    return current_state_index


# a function to find the index of a composite state in the product matrix.
# Input: indices i and j, number of states. Output: composite index.
def composite_index(i, j, num_states):
    return (i - 1) * num_states + j


# a function to construct the 2-site transition matrix by taking the "product" of two 1-site matrix.
# Input: a 1-site transition matrix. Output: a 2-site transition matrix with state space size squared.
def matrix_product(rate_matrix):
    num_states = rate_matrix.shape[0]
    product_matrix = np.zeros((num_states ** 2, num_states ** 2))
    for i in range(num_states):
        for j in range(num_states):
            for k in range(num_states):
                product_matrix[composite_index(i, k, num_states), composite_index(i, j, num_states)] = rate_matrix[k, j]
                product_matrix[composite_index(k, j, num_states), composite_index(i, j, num_states)] = rate_matrix[k, i]
    for i in range(num_states):
        for j in range(num_states):
            product_matrix[composite_index(i, j, num_states), composite_index(i, j, num_states)] = rate_matrix[i, i] + rate_matrix[j, j]
    return product_matrix


# a function to sample from a stationery distribution.
# Input: stationery distribution and the state space. Output: a state.
def stationery_dist_sampler(stationery_dist, all_states):
    return random.choices(all_states, weights=stationery_dist)


# Generate the variation at a site given a phylogenetic tree
def build_empty_seq_dict(tree):
    seq_dict = {}
    leaf_nodes = tree.get_leaf_names()
    for i in leaf_nodes:
        seq_dict[i] = []
    return seq_dict


def add_variation(tree, root_state, rate_matrix, seq_dict):
    if tree.get_children():
        for subtree in tree.get_children():
            branch_length = subtree.dist
            internal_state = ending_state(root_state, rate_matrix, branch_length)
            add_variation(subtree, internal_state, rate_matrix, seq_dict)
    else:
        node_name = tree.name
        seq_dict[node_name].append(root_state)


def test_character_simulation():
    t = Tree("(A:1,(B:1,(E:1,D:1):0.5):0.5);")
    rate_matrix = np.array([[-1, 1], [1, -1]])
    root_state = 0
    seq_dict = build_empty_seq_dict(t)
    for i in range(10):
        add_variation(t, root_state, rate_matrix, seq_dict)
    print(seq_dict)