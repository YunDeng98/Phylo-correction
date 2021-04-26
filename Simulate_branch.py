import Phylo_util
import random
import numpy as np


def branch_pair(model, t: float) -> list:
    stationery_dist = model.stationery_dist
    n = model.rate_matrix.shape[0]
    all_states = [i for i in range(n)]
    start_index = random.choices(population=all_states, weights=stationery_dist, k=1)[0]
    end_index = Phylo_util.ending_state(start_index, model, t)
    return [start_index, end_index]


def simulate_branches(rate_matrix, stationery_dist, t: float, n: int) -> list:
    model = Phylo_util.substitution_model(rate_matrix, stationery_dist)
    pair_dist = []
    for i in range(n):
        pair_dist.append(branch_pair(model, t))
    return pair_dist


rate_matrix = np.loadtxt('WAG_matrix.txt')
stationery_dist = np.loadtxt('WAG_stationery_dist.txt')
print(simulate_branches(rate_matrix, stationery_dist, 0.1, 100))