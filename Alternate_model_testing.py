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


def solve_stationery_dist(rate_matrix):
    eigvals, eigvecs = np.linalg.eig(rate_matrix.transpose())
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    eigvals = np.abs(eigvals)
    index = np.argmin(eigvals)
    stationery_dist = eigvecs[:, index]
    stationery_dist = stationery_dist / sum(stationery_dist)
    return stationery_dist


def build_sequence_alignment(tree, rate_matrix, stationery_dist, n):
    seq_dict = Phylo_simulator.build_empty_seq_dict(tree)
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
    permute_order = np.array([i for i in range(10, n)] + [i for i in range(10)])
    for i in range(n):
        mid_rate_matrix[:, i] = rate_matrix[:, permute_order[i]]
    for i in range(n):
        new_rate_matrix[i, :] = mid_rate_matrix[permute_order[i], :]
    for i in range(n):
        new_stationery_dist[i] = stationery_dist[permute_order[i]]
    return new_rate_matrix, new_stationery_dist


def generate_fitness_landscape(n, variance):
    fitness_landscape = np.random.normal(0, variance, n)
    return fitness_landscape


def modify_rate_matrix(rate_matrix, fitness_landscape):
    n = rate_matrix.shape[0]
    new_rate_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            new_rate_matrix[i, j] = np.exp(fitness_landscape[i] - fitness_landscape[j]) * rate_matrix[i, j]
    for i in range(n):
        new_rate_matrix[i, i] = 0
        new_rate_matrix[i, i] = -sum(new_rate_matrix[i, :])
    return new_rate_matrix
