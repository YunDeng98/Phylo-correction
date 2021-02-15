import numpy as np
import scipy as sp
import pandas as pd
import random
import ete3
from ete3 import Tree, TreeNode
from scipy import linalg as SA
import Phylo_simulator
from Phylo_simulator import matrix_product


def branch_matrix_exponential(rate_matrix, t):
    branch_rate_matrix = t * rate_matrix
    return SA.expm(branch_rate_matrix)


def log_inner_product(rate_matrix_col, log_likelihood):
    n = len(log_likelihood)
    max_log = np.max(log_likelihood)
    log_likelihood -= max_log
    inner_product = 0
    for i in range(n):
        inner_product += rate_matrix_col[i] * np.exp(log_likelihood[i])
    return np.log(inner_product) + max_log


def tree_log_likelihood(tree, seq_dict, n):
    log_likelihood_dict = {}
    stationery_dist = np.loadtxt('WAG_stationery_dist.txt')
    rate_matrix = np.loadtxt('WAG_matrix.txt')
    num_states = len(stationery_dist)
    for leaf in tree:
        state = seq_dict[leaf.name][n]
        log_likelihood_dict[leaf] = [-np.inf for i in range(num_states)]
        log_likelihood_dict[leaf][state] = 0
    fill_log_likelihood(tree, log_likelihood_dict, rate_matrix)
    return log_inner_product(stationery_dist, log_likelihood_dict[tree])


def log_likelihood_update(log_likelihood_1, log_likelihood_2, transition_matrix_1, transition_matrix_2):
    num_states = len(log_likelihood_1)
    new_log_likelihood = np.zeros(num_states)
    for i in range(num_states):
        log_product_1 = log_inner_product(transition_matrix_1[i, ], log_likelihood_1)
        log_product_2 = log_inner_product(transition_matrix_2[i, ], log_likelihood_2)
        new_log_likelihood[i] = log_product_1 + log_product_2
    return new_log_likelihood


def fill_log_likelihood(node, log_likelihood_dict, rate_matrix):
    if node in log_likelihood_dict:
        return log_likelihood_dict[node]
    else:
        child_node_1, child_node_2 = node.get_children()
        t_1 = child_node_1.dist
        t_2 = child_node_2.dist
        transition_matrix_1 = branch_matrix_exponential(rate_matrix, t_1)
        transition_matrix_2 = branch_matrix_exponential(rate_matrix, t_2)
        log_likelihood_1 = fill_log_likelihood(child_node_1, log_likelihood_dict, rate_matrix)
        log_likelihood_2 = fill_log_likelihood(child_node_2, log_likelihood_dict, rate_matrix)
        new_log_likelihood = log_likelihood_update(log_likelihood_1, log_likelihood_2, transition_matrix_1, transition_matrix_2)
        log_likelihood_dict[node] = new_log_likelihood
        return new_log_likelihood


