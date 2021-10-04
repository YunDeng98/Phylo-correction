import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
import logging
from typing import Optional
from scipy.stats import norm

from . import RateMatrix, train_quantization, solve_stationery_dist


def normalized(Q):
    pi = solve_stationery_dist(Q)
    mutation_rate = pi @ -np.diag(Q)
    return Q / mutation_rate


class RateMatrixLearner:
    def __init__(
        self,
        frequency_matrices: str,
        output_dir: str,
        stationnary_distribution: str,
        device: str,
        mask: str = None,
        frequency_matrices_sep="\s",
        rate_matrix_parameterization="pande_reversible",
        use_cached: bool = False,
        initialization: Optional[np.array] = None,
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
        self.device = device
        self.use_cached = use_cached
        self.initialization = initialization

    def train(
        self,
        lr=1e-1,
        num_epochs=2000,
        do_adam: bool = True,
    ):
        logger = logging.getLogger("phylo_correction.ratelearner")
        logger.info(f"Starting, outdir: {self.output_dir}")

        torch.manual_seed(0)
        device = self.device
        output_dir = self.output_dir
        use_cached = self.use_cached
        initialization = self.initialization

        # Create experiment directory
        if os.path.exists(output_dir) and use_cached:
            # logger.info(f"Skipping. Cached ratelearner results at {output_dir}")
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

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
        mask_mat = torch.tensor(mask_mat, dtype=torch.float)
        self.mask_mat = mask_mat

        pi_requires_grad = pi_path is None
        self.mat_module = RateMatrix(
            num_states=self.n_states,
            mode=self.rate_matrix_parameterization,
            pi=self.pi,
            pi_requires_grad=pi_requires_grad,
            initialization=initialization,
            mask=mask_mat,
        ).to(device=device)

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

    def compute_fisher_information(self):
        """Computes the Fisher information corresponding to a world
        where observations have quantized branch lengths.

        Returns
        -------
        tensor
            2d Hessian (with respect to the vectorized rate matrix)
        """
        rate_module = self.mat_module
        quantized_dataset = self.quantized_data
        branches, cmat = quantized_dataset.tensors[0], quantized_dataset.tensors[1]

        assert self.mat_module.mode == "pande_reversible"

        Q = rate_module()
        n_states = Q.shape[-1]
        # WE ONLY DIFFERENTIATE WRT STRICT TRIANGULAR UPPER RATE MATRIX ELEMENTS
        Q1d = Q[torch.triu_indices(n_states, n_states, offset=1).unbind()]

        device = Q.device
        branch_length = branches.to(device=device)
        cmat = cmat.to(device=device)

        def get_score(mat1d):
            """Computes sum of scores given upper triangular rate matrix indices

            Parameters
            ----------
            mat1d :
                upper triangular rate matrix (1d)

            Returns
            -------
            scalar
                sum of scores
            """            
            mat = torch.zeros(n_states, n_states)
            mat[torch.triu_indices(n_states, n_states, offset=1).unbind()] = mat1d
            mat = mat + mat.T
            # Does not change values on upper diagonal
            mat = mat - torch.diag(mat.sum(0))
            branch_length_ = branch_length
            mats = torch.log(torch.matrix_exp(branch_length_[:, None, None] * mat))
            mats = mats * cmat
            # Observed Fisher is sum of the score Hessians
            # Which is the same as the Hessian of the sum
            return mats.sum()

        return -torch.autograd.functional.hessian(get_score, Q1d)


    def get_confidence_intervals(self, alpha):
        """Returns marginal confidence intervals for each matrix coefficient"""
        fisher_info = self.compute_fisher_information()
        q = self.mat_module().detach()
        num_states = q.shape[-1]
        # q_ = q[
        #     torch.triu_indices(num_states, num_states, offset=1).unbind()
        # ]
        # q1d = q_.reshape(-1)
        n_obs = self.quantized_data.tensors[1].sum().item()

        # From Jn estimate to J1
        fisher_info = fisher_info / n_obs
        inv_fisher_info = torch.inverse(fisher_info)
        variances = torch.diag(inv_fisher_info)

        all_variances = torch.zeros(num_states, num_states)
        all_variances[torch.triu_indices(num_states, num_states, offset=1).unbind()] = variances
        all_variances = all_variances + all_variances.T
        all_variances += torch.diag(all_variances).sum(1)  
        # variance of diagonal term is sum of row variances

        all_qs = q
        # This is the Hessian wrt to the vectorized features
        zis = norm.ppf(1 - alpha / 2)
        delta_vals = all_variances.sqrt() / np.sqrt(n_obs) * zis
        vinf2d = all_qs - delta_vals
        vsup2d = all_qs + delta_vals
        return vinf2d, vsup2d

    def process_results(self):
        learned_matrix_path = os.path.join(self.output_dir, "learned_matrix.txt")
        Q = self.Qfinal.detach().cpu().numpy()
        np.savetxt(learned_matrix_path, Q)
        os.system(f"chmod 555 {learned_matrix_path}")

        normalized_learned_matrix_path = os.path.join(
            self.output_dir, "learned_matrix_normalized.txt"
        )
        np.savetxt(normalized_learned_matrix_path, normalized(Q))
        os.system(f"chmod 555 {normalized_learned_matrix_path}")

        df_res_filepath = os.path.join(self.output_dir, "training_df.pickle")
        self.df_res.to_pickle(df_res_filepath)
        os.system(f"chmod 555 {df_res_filepath}")

        FT_SIZE = 13
        fig, axes = plt.subplots(figsize=(5, 4))
        self.df_res.loss.plot()
        plt.xscale("log")
        plt.ylabel("Negative likelihood", fontsize=FT_SIZE)
        plt.xlabel("# of iterations", fontsize=FT_SIZE)
        plt.tight_layout()
        figpath = os.path.join(self.output_dir, "training_plot.pdf")
        plt.savefig(figpath)
        os.system(f"chmod 555 {figpath}")
