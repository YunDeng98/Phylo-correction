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
    floored_log_likelihood = log_likelihood - max_log
    inner_product = np.dot(rate_matrix_col, np.exp(floored_log_likelihood))
    return np.log(inner_product) + max_log


def tree_log_likelihood(tree, seq_dict, rate_matrix, stationery_dist):
    root_log_likelihood = node_log_likelihood(tree, rate_matrix, seq_dict)
    return log_inner_product(stationery_dist, root_log_likelihood)


def log_likelihood_update(log_likelihood_1, log_likelihood_2, transition_matrix_1, transition_matrix_2):
    num_states = len(log_likelihood_1)
    new_log_likelihood = np.zeros(num_states)
    for i in range(num_states):
        log_product_1 = log_inner_product(transition_matrix_1[i, :], log_likelihood_1)
        log_product_2 = log_inner_product(transition_matrix_2[i, :], log_likelihood_2)
        new_log_likelihood[i] = log_product_1 + log_product_2
    return new_log_likelihood


def node_log_likelihood(node, rate_matrix, seq_dict):
    if node.is_leaf():
        num_states = rate_matrix.shape[0]
        state = seq_dict[node.name]
        new_log_likelihood = [-np.inf for i in range(num_states)]
        new_log_likelihood[state] = 0
        return new_log_likelihood
    else:
        child_node_1, child_node_2 = node.get_children()
        t_1 = child_node_1.dist
        t_2 = child_node_2.dist
        transition_matrix_1 = branch_matrix_exponential(rate_matrix, t_1)
        transition_matrix_2 = branch_matrix_exponential(rate_matrix, t_2)
        log_likelihood_1 = node_log_likelihood(child_node_1, rate_matrix, seq_dict)
        log_likelihood_2 = node_log_likelihood(child_node_2, rate_matrix, seq_dict)
        new_log_likelihood = log_likelihood_update(log_likelihood_1, log_likelihood_2, transition_matrix_1, transition_matrix_2)
        return new_log_likelihood


