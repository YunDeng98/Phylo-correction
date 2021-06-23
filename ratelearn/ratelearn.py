import numpy as np
import pandas as pd
import torch
import os
import argparse
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import logging

from ratelearn import RateMatrix, train_quantization


def parse_args():
    parser = argparse.ArgumentParser("Data2TabShop")
    parser.add_argument("--frequency_matrices", type=str)
    parser.add_argument("--stationnary_distribution", type=str, default=None)
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--do_adam", type=bool, default=True)
    return parser.parse_args()


def get_branch_to_mat(file_path, sep="\s"):
    matrices_file = pd.read_csv(
        file_path, sep=sep, header=None, index_col=None
    ).reset_index()
    where_branch = matrices_file.isna().any(axis=1)
    branch_indices = where_branch[where_branch].index

    separation_between_branches = np.unique(branch_indices[1:] - branch_indices[:-1])
    assert len(separation_between_branches) == 1, separation_between_branches
    n_features = len(matrices_file.columns)
    assert n_features == separation_between_branches[0] - 1

    branch_to_mat = dict()
    branches = []
    mats = []
    for branch_idx in branch_indices:
        branch_len = matrices_file.iloc[branch_idx, 0]
        mat = matrices_file.loc[branch_idx + 1 : branch_idx + n_features].values
        # branch_to_mat[branch_len] = mat
        branches.append(branch_len)
        mats.append(mat)
    # qtimes = torch.tensor(list(branch_to_mat.keys()))
    # cmats = torch.tensor(list(branch_to_mat.values()))
    qtimes = torch.tensor(branches)
    cmats = torch.tensor(mats)
    quantized_data = TensorDataset(qtimes, cmats)
    return quantized_data, n_features


args = parse_args()
if __name__ == "__main__":
    logging.warn(
        """
            This script provides a strategy to learn the rate matrix using GD.
            If necessary, a backtracking line search will be implemented to automatically set the learning rate.
            Please check the convergence of the algorithm after training (see `training_plot.pdf`).
            If you see erratic behaviors consider tuning the learning rate with the argument --lr.
        """
    )
    output_path = args.output_dir
    if os.path.exists(output_path):
        raise ValueError("Please provide a nonexisting experiment path")
    os.makedirs(output_path)

    quantized_data, n_states = get_branch_to_mat(args.frequency_matrices)
    pi_path = args.stationnary_distribution
    if pi_path is not None:
        pi = pd.read_csv(pi_path, header=None, index_col=None).values.squeeze()
    else:
        pi = np.ones(n_states)
        pi = pi / pi.sum()
    pi = torch.tensor(pi).float()

    mask_path = args.mask
    if mask_path is not None:
        mask_mat = pd.read_csv(mask_path, sep="\s", header=None, index_col=None).values
    else:
        mask_mat = np.ones((n_states, n_states))
    mask_mat = torch.tensor(mask_mat)

    mat_module = RateMatrix(
        num_states=n_states,
        mode="pande_reversible",
        pi=pi,
        pi_requires_grad=True,
    ).cuda()
    if args.do_adam:
        optim = torch.optim.Adam(params=mat_module.parameters(), lr=args.lr)
    else:
        optim = torch.optim.SGD(params=mat_module.parameters(), lr=args.lr)
    df_res, Q = train_quantization(
        rate_module=mat_module,
        quantized_dataset=quantized_data,
        num_epochs=2000,
        Q_true=None,
        optimizer=optim,
    )
    np.savetxt(
        os.path.join(output_path, "learned_matrix.txt"), Q.detach().cpu().numpy()
    )

    df_res.to_pickle(os.path.join(output_path, "training_df.pickle"))
    FT_SIZE = 13
    fig, axes = plt.subplots(figsize=(5, 4))
    df_res.loss.plot()
    plt.xscale("log")
    plt.ylabel("Negative likelihood", fontsize=FT_SIZE)
    plt.xlabel("# of iterations", fontsize=FT_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "training_plot.pdf"))
