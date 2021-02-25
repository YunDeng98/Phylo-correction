import numpy as np


def log_inner_product(rate_matrix_col, log_likelihood):
    n = len(log_likelihood)
    max_log = np.max(log_likelihood)
    log_likelihood -= max_log
    inner_product = np.dot(rate_matrix_col, np.exp(log_likelihood))
    return np.log(inner_product) + max_log


def tree_log_likelihood(tree, seq_dict, model):
    root_log_likelihood = node_log_likelihood(tree, model, seq_dict)
    return log_inner_product(model.stationery_dist, root_log_likelihood)


def log_likelihood_update(log_likelihood_1, log_likelihood_2, transition_matrix_1, transition_matrix_2):
    num_states = len(log_likelihood_1)
    new_log_likelihood = np.zeros(num_states)
    max_log_1 = np.max(log_likelihood_1)
    max_log_2 = np.max(log_likelihood_2)
    floored_log_likelihood_1 = log_likelihood_1 - max_log_1
    floored_log_likelihood_2 = log_likelihood_2 - max_log_2
    sum_1 = np.dot(transition_matrix_1, np.exp(floored_log_likelihood_1))
    sum_2 = np.dot(transition_matrix_2, np.exp(floored_log_likelihood_2))
    new_log_likelihood = np.log(sum_1) + np.log(sum_2) + max_log_1 + max_log_2
    return new_log_likelihood


def node_log_likelihood(node, model, seq_dict):
    if node.is_leaf():
        num_states = model.rate_matrix.shape[0]
        state = seq_dict[node.name]
        new_log_likelihood = [-np.inf for i in range(num_states)]
        new_log_likelihood[state] = 0
        return new_log_likelihood
    else:
        child_node_1, child_node_2 = node.get_children()
        t_1 = child_node_1.dist
        t_2 = child_node_2.dist
        transition_matrix_1 = model.expm(t_1)
        transition_matrix_2 = model.expm(t_2)
        log_likelihood_1 = node_log_likelihood(child_node_1, model, seq_dict)
        log_likelihood_2 = node_log_likelihood(child_node_2, model, seq_dict)
        new_log_likelihood = log_likelihood_update(log_likelihood_1, log_likelihood_2, transition_matrix_1, transition_matrix_2)
        return new_log_likelihood
