import numpy as np
import scipy as sp
import random
from scipy import linalg as SA
import ete3
from ete3 import Tree, TreeNode
import Phylo_simulator
import Felsenstein_algorithm
import Fast_Felsenstein_algorithm
import Alternate_model_testing
import ngesh
import Phylo_util
from timeit import default_timer as timer


def fast_vs_standard_felsenstein():
    rate_matrix = np.loadtxt('WAG_matrix.txt')
    stationery_dist = np.loadtxt('WAG_stationery_dist.txt')
    product_rate_matrix = Phylo_simulator.matrix_product(rate_matrix)
    product_stationery_dist = Phylo_simulator.stationery_dist_product(stationery_dist)
    null_model = Phylo_util.substitution_model(product_rate_matrix, product_stationery_dist)
    tree = Tree("test_tree.nw", format=1)
    seq_dict = Alternate_model_testing.build_sequence_alignment(tree, product_rate_matrix, product_stationery_dist, 1)
    start = timer()
    l1 = Fast_Felsenstein_algorithm.tree_log_likelihood(tree, seq_dict, 0, null_model)
    end = timer()
    print(end - start)
    start = timer()
    l2 = Felsenstein_algorithm.tree_log_likelihood(tree, seq_dict, 0, product_rate_matrix, product_stationery_dist)
    end = timer()
    print(end - start)
    print(l1, l2)


def LR_test():
    rate_matrix = np.loadtxt('WAG_matrix.txt')
    stationery_dist = np.loadtxt('WAG_stationery_dist.txt')
    new_rate_matrix = np.loadtxt('WAG_new_matrix.txt')
    new_stationery_dist = np.repeat(0.05, 20)
    product_rate_matrix = Phylo_simulator.matrix_product(rate_matrix)
    product_stationery_dist = Phylo_simulator.stationery_dist_product(stationery_dist)
    new_product_rate_matrix = Phylo_simulator.matrix_product(new_rate_matrix)
    new_product_stationery_dist = np.repeat(0.0025, 400)
    null_model = Phylo_util.substitution_model(product_rate_matrix, product_stationery_dist)
    new_model = Phylo_util.substitution_model(new_product_rate_matrix, new_product_stationery_dist)
    tree = Tree('tall_tree.nw', format=1)
    for i in range(1000):
        seq_dict = Phylo_util.build_sequence_alignment(tree, null_model)
        l1 = Fast_Felsenstein_algorithm.tree_log_likelihood(tree, seq_dict, null_model)
        l2 = Fast_Felsenstein_algorithm.tree_log_likelihood(tree, seq_dict, new_model)
        print(l1, l2, l1 - l2)


def interaction_model_test(scale):
    rate_matrix = np.loadtxt('WAG_matrix.txt')
    product_rate_matrix = Phylo_simulator.matrix_product(rate_matrix)
    stationery_dist = np.loadtxt('WAG_stationery_dist.txt')
    product_stationery_dist = Phylo_simulator.stationery_dist_product(stationery_dist)
    num_states = product_rate_matrix.shape[0]
    fitness_landscape = Alternate_model_testing.generate_fitness_landscape(num_states, scale)
    new_product_rate_matrix = Alternate_model_testing.modify_rate_matrix(product_rate_matrix, fitness_landscape)
    new_product_stationery_dist = Alternate_model_testing.solve_stationery_dist(new_product_rate_matrix)
    null_model = Phylo_util.substitution_model(product_rate_matrix, product_stationery_dist)
    new_model = Phylo_util.substitution_model(new_product_rate_matrix, new_product_stationery_dist)
    tree = Tree('tall_tree.nw', format=1)
    for i in range(50):
        print(i)
        seq_dict = Phylo_util.build_sequence_alignment(tree, new_model)
        l1 = Fast_Felsenstein_algorithm.tree_log_likelihood(tree, seq_dict, null_model)
        l2 = Fast_Felsenstein_algorithm.tree_log_likelihood(tree, seq_dict, new_model)
        print(l1, l2, l1 - l2)


def test_alignment(tree, stationery_dist):
    seq_dict = {}
    all_states = [i for i in range(len(stationery_dist))]
    consensus_state = random.choices(population=all_states, weights=stationery_dist, k=1)[0]
    for i in tree.get_leaf_names():
        seq_dict[i] = consensus_state
    return seq_dict


def fixed_interaction():
    rate_matrix = np.loadtxt('WAG_matrix.txt')
    product_rate_matrix = Phylo_util.matrix_product(rate_matrix)
    stationery_dist = np.loadtxt('WAG_stationery_dist.txt')
    product_stationery_dist = Phylo_util.stationery_dist_product(stationery_dist)
    new_product_rate_matrix = np.loadtxt('interaction_model.txt')
    new_product_stationery_dist = Phylo_util.solve_stationery_dist(new_product_rate_matrix)
    null_model = Phylo_util.substitution_model(product_rate_matrix, product_stationery_dist)
    new_model = Phylo_util.substitution_model(new_product_rate_matrix, new_product_stationery_dist)
    tree = Tree('test_tree.nw', format=1)
    for i in range(10):
        seq_dict_1 = Phylo_util.build_sequence_alignment(tree, null_model)
        seq_dict_2 = Phylo_util.build_sequence_alignment(tree, new_model)
        print('seq_dict_1: ', seq_dict_1)
        print('seq_dict_2: ', seq_dict_2)
        l1 = Fast_Felsenstein_algorithm.tree_log_likelihood(tree, seq_dict_1, null_model)
        l2 = Fast_Felsenstein_algorithm.tree_log_likelihood(tree, seq_dict_2, null_model)
        print(l1, l2, l1 - l2)


def random_check():
    rate_matrix = np.loadtxt('WAG_matrix.txt')
    product_rate_matrix = Phylo_util.matrix_product(rate_matrix)
    stationery_dist = np.loadtxt('WAG_stationery_dist.txt')
    product_stationery_dist = Phylo_util.stationery_dist_product(stationery_dist)
    new_product_rate_matrix = np.loadtxt('interaction_model.txt')
    new_product_stationery_dist = Phylo_util.solve_stationery_dist(new_product_rate_matrix)
    null_model = Phylo_util.substitution_model(product_rate_matrix, product_stationery_dist)
    new_model = Phylo_util.substitution_model(new_product_rate_matrix, new_product_stationery_dist)
    tree = Tree('test_tree.nw', format=1)
    seq_dict_0 = test_alignment(tree, product_stationery_dist)
    for i in range(10):
        seq_dict_1 = Phylo_util.build_sequence_alignment(tree, null_model)
        print('seq_dict_1: ', seq_dict_0)
        print('seq_dict_2: ', seq_dict_1)
        l1 = Fast_Felsenstein_algorithm.tree_log_likelihood(tree, seq_dict_0, null_model)
        l2 = Fast_Felsenstein_algorithm.tree_log_likelihood(tree, seq_dict_1, null_model)
        print(l1, l2, l1 - l2)


def cache_test():
    rate_matrix = np.loadtxt('WAG_matrix.txt')
    stationery_dist = np.loadtxt('WAG_stationery_dist.txt')
    new_rate_matrix = np.loadtxt('WAG_new_matrix.txt')
    new_stationery_dist = np.repeat(0.05, 20)
    null_model = Phylo_util.substitution_model(rate_matrix, stationery_dist)
    new_model = Phylo_util.substitution_model(new_rate_matrix, new_stationery_dist)
    print(null_model.expm(1) - new_model.expm(1))

LR_test()
