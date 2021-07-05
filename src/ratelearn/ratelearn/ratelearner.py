import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

from . import RateMatrix, train_quantization


class RateMatrixLearner:
    def __init__(
        self,
        frequency_matrices: str,
        output_dir: str,
        stationnary_distribution: str,
        mask: str = None,
        frequency_matrices_sep="\s",
        rate_matrix_parameterization="pande_reversible",
    ):
        self.frequency_matrices = frequency_matrices
        self.frequency_matrices_sep = frequency_matrices_sep
        self.output_dir = output_dir
        self.stationnary_distribution = stationnary_distribution
        self.mask = mask
        self.rate_matrix_parameterization = rate_matrix_parameterization
        self.lr = None
        self.do_adam = None
        self.df_res = None
        self.Qfinal = None
        self.trained_ = False

        # Create experiment directory
        if os.path.exists(self.output_dir):
            raise ValueError("Please provide a nonexisting experiment path")
        os.makedirs(self.output_dir)

        # Open frequency matrices
        self.quantized_data, self.n_states = self.get_branch_to_mat()

        # Open stationnary distribution if necessary
        pi_path = self.stationnary_distribution
        if pi_path is not None:
            pi = pd.read_csv(pi_path, header=None, index_col=None).values.squeeze()
        else:
            pi = np.ones(self.n_states)
            pi = pi / pi.sum()
        self.pi = torch.tensor(pi).float()

        mask_path = self.mask
        if mask_path is not None:
            mask_mat = pd.read_csv(
                mask_path, sep="\s", header=None, index_col=None
            ).values
        else:
            mask_mat = np.ones((self.n_states, self.n_states))
        self.mask_mat = torch.tensor(mask_mat)

        pi_requires_grad = pi_path is None
        self.mat_module = RateMatrix(
            num_states=self.n_states,
            mode=self.rate_matrix_parameterization,
            pi=self.pi,
            pi_requires_grad=pi_requires_grad,
        )#.cuda()

    def train(
        self,
        lr=1e-1,
        num_epochs=2000,
        do_adam: bool = True,
    ):
        self.lr = lr
        self.do_adam = do_adam

        if self.do_adam:
            optim = torch.optim.Adam(params=self.mat_module.parameters(), lr=self.lr)
        else:
            optim = torch.optim.SGD(params=self.mat_module.parameters(), lr=self.lr)

        df_res, Q = train_quantization(
            rate_module=self.mat_module,
            quantized_dataset=self.quantized_data,
            num_epochs=num_epochs,
            Q_true=None,
            optimizer=optim,
        )
        self.df_res = df_res
        self.Qfinal = Q
        self.trained = True
        self.process_results()

    def get_branch_to_mat(self):
        sep = self.frequency_matrices_sep
        matrices_file = pd.read_csv(
            self.frequency_matrices, sep=sep, header=None, index_col=None
        ).reset_index()
        where_branch = matrices_file.isna().any(axis=1)
        branch_indices = where_branch[where_branch].index

        separation_between_branches = np.unique(
            branch_indices[1:] - branch_indices[:-1]
        )
        assert len(separation_between_branches) == 1, separation_between_branches
        n_features = len(matrices_file.columns)
        assert n_features == separation_between_branches[0] - 1

        branches = []
        mats = []
        for branch_idx in branch_indices:
            branch_len = matrices_file.iloc[branch_idx, 0]
            mat = matrices_file.loc[branch_idx + 1 : branch_idx + n_features].values
            branches.append(branch_len)
            mats.append(mat)
        qtimes = torch.tensor(branches)
        cmats = torch.tensor(mats)
        quantized_data = TensorDataset(qtimes, cmats)
        return quantized_data, n_features

    def process_results(self):
        np.savetxt(
            os.path.join(self.output_dir, "learned_matrix.txt"),
            self.Qfinal.detach().cpu().numpy(),
        )

        self.df_res.to_pickle(os.path.join(self.output_dir, "training_df.pickle"))
        FT_SIZE = 13
        fig, axes = plt.subplots(figsize=(5, 4))
        self.df_res.loss.plot()
        plt.xscale("log")
        plt.ylabel("Negative likelihood", fontsize=FT_SIZE)
        plt.xlabel("# of iterations", fontsize=FT_SIZE)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "training_plot.pdf"))