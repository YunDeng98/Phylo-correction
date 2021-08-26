import os
import numpy as np
import pandas as pd
import torch
import logging
import sys
sys.path.append("../")
import Phylo_util


def normalized(Q):
    pi = Phylo_util.solve_stationery_dist(Q)
    mutation_rate = pi @ -np.diag(Q)
    return Q / mutation_rate


class JTT:
    def __init__(
        self,
        frequency_matrices: str,
        output_dir: str,
        mask: str = None,
        frequency_matrices_sep="\s",
        use_cached: bool = False,
    ):
        self.frequency_matrices = frequency_matrices
        self.output_dir = output_dir
        self.mask = mask
        self.frequency_matrices_sep = frequency_matrices_sep
        self.use_cached = use_cached

    def train(
        self,
    ):
        frequency_matrices = self.frequency_matrices
        output_dir = self.output_dir
        mask = self.mask
        use_cached = self.use_cached

        logger = logging.getLogger("phylo_correction.counting")

        # Create experiment directory
        if os.path.exists(output_dir) and use_cached:
            logger.info(f"Skipping. Cached counting JTT results at {output_dir}")
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Open frequency matrices
        self.quantized_data, self.n_states = self.get_branch_to_mat()

        mask_path = self.mask
        if mask_path is not None:
            mask_mat = pd.read_csv(
                mask_path, sep="\s", header=None, index_col=None
            ).values
        else:
            mask_mat = np.ones((self.n_states, self.n_states))
        self.mask_mat = torch.tensor(mask_mat)

        (qtimes, cmats), n_states = self.quantized_data, self.n_states

        # Compute frequency matrix ignoring branch lengths
        F = cmats.sum(axis=0)
        # Coalesce transitions a->b and b->a together
        F = (F + np.transpose(F)) / 2.0
        # Compute mutabilities
        M = F.sum(axis=1)
        # JTT estimator
        res = F / (M[:, None] + 1e-16)
        np.fill_diagonal(res, -((1.0 - np.eye(n_states)) * res).sum(axis=1))
        # Some guess of the scaling constant;
        # doesn't matter if we normalize later to mutation rate 1.
        lam = 1.0 / (np.median(qtimes) + 1e-16)
        res = res * lam

        learned_matrix_path = os.path.join(self.output_dir, "learned_matrix.txt")
        np.savetxt(learned_matrix_path, res)
        os.system(f"chmod 555 {learned_matrix_path}")

        normalized_learned_matrix_path = os.path.join(self.output_dir, "learned_matrix_normalized.txt")
        np.savetxt(normalized_learned_matrix_path, normalized(res))
        os.system(f"chmod 555 {normalized_learned_matrix_path}")

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
        qtimes = np.array(branches)
        cmats = np.array(mats)
        quantized_data = (qtimes, cmats)
        return quantized_data, n_features
