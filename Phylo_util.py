import numpy as np
import random
from functools import lru_cache


class substitution_model:

    def __init__(self, rate_matrix, stationery_dist):
        self.rate_matrix = rate_matrix
        self.stationery_dist = stationery_dist
        self.S = self.symmetrize_rate_matrix(rate_matrix, stationery_dist)
        self.P1, self.P2 = self.diagonal_stationery_matrix(stationery_dist)
        self.D, self.U = np.linalg.eigh(self.S)

    @staticmethod
    def symmetrize_rate_matrix(rate_matrix, stationery_dist):
        n = rate_matrix.shape[0]
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                S[i, j] = rate_matrix[i, j] * np.sqrt(stationery_dist[i]) / np.sqrt(stationery_dist[j])
        return S

    @staticmethod
    def diagonal_stationery_matrix(stationery_dist):
        n = len(stationery_dist)
        P1 = np.diag(np.sqrt(stationery_dist))
        P2 = np.diag(np.sqrt(1 / stationery_dist))
        return P1, P2

    @lru_cache(maxsize=None)
    def expm(self, t):
        n = self.D.shape[0]
        exp_D = np.diag(np.exp(t * self.D))
        exp_S = np.dot(np.dot(self.U, exp_D), self.U.transpose())
        exp_R = np.dot(np.dot(self.P2, exp_S), self.P1)
        return exp_R


def ending_state(initial_state_index, model, t):
    exp_R = model.expm(t)
    n = model.rate_matrix.shape[0]
    all_states = [i for i in range(n)]
    ending_state_index = random.choices(population=all_states, weights=exp_R[initial_state_index, :], k=1)[0]
    return ending_state_index


def ending_state_without_expm(initial_state_index, model, t):
    """
    Computes the next state by sampling the whole intermediate history.
    I.e., we just sample from exponential distributions; we never compute the
    matrix exponential.
    """
    rate_matrix = model.rate_matrix
    n = rate_matrix.shape[0]
    curr_state_index = initial_state_index
    current_t = 0  # We simulate the process starting from time 0.
    while True:
        # See when the next transition happens
        waiting_time = np.random.exponential(1.0 / -rate_matrix[curr_state_index, curr_state_index])
        current_t += waiting_time
        if current_t >= t:
            # We reached the end of the process
            return curr_state_index
        # Update the curr_state_index
        weights =\
            list(rate_matrix[
                curr_state_index,
                :curr_state_index]) +\
            list(rate_matrix[
                curr_state_index,
                (curr_state_index + 1):])
        assert(len(weights) == n - 1)
        new_state_index = random.choices(
            population=range(n - 1),
            weights=weights,
            k=1)[0]
        # Because new_state_index is in [0, n - 2], we must map it back to [0, n - 1].
        if new_state_index >= curr_state_index:
            new_state_index += 1
        curr_state_index = new_state_index


def add_variation(tree, root_state, model, seq_dict):
    if tree.get_children():
        for subtree in tree.get_children():
            branch_length = subtree.dist
            internal_state = ending_state(root_state, model, branch_length)
            add_variation(subtree, internal_state, model, seq_dict)
    else:
        node_name = tree.name
        seq_dict[node_name] = root_state


def build_sequence_alignment(tree, model):
    seq_dict = {}
    stationery_dist = model.stationery_dist
    all_states = [i for i in range(len(stationery_dist))]
    root_state = random.choices(population=all_states, weights=stationery_dist, k=1)[0]
    add_variation(tree, root_state, model, seq_dict)
    return seq_dict


def composite_index(i, j, num_states):
    return (i - 1) * num_states + j


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


def stationery_dist_product(stationery_dist):
    num_states = len(stationery_dist)
    product_stationery_dist = np.zeros(num_states ** 2)
    for i in range(num_states):
        for j in range(num_states):
            product_stationery_dist[composite_index(i, j, num_states)] = stationery_dist[i]*stationery_dist[j]
    return product_stationery_dist


def solve_stationery_dist(rate_matrix):
    eigvals, eigvecs = np.linalg.eig(rate_matrix.transpose())
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    eigvals = np.abs(eigvals)
    index = np.argmin(eigvals)
    stationery_dist = eigvecs[:, index]
    stationery_dist = stationery_dist / sum(stationery_dist)
    return stationery_dist



