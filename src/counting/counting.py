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
        use_cached: bool = False,  # TODO: Currently not used, since the method is so fast.
        ipw: bool = False,
    ):
        self.frequency_matrices = frequency_matrices
        self.output_dir = output_dir
        self.mask = mask
        self.frequency_matrices_sep = frequency_matrices_sep
        self.use_cached = use_cached
        self.ipw = ipw

    def train(
        self,
    ):
        logger = logging.getLogger("phylo_correction.counting")
        logger.info(f"Starting, outdir: {self.output_dir}")

        frequency_matrices = self.frequency_matrices
        output_dir = self.output_dir
        mask = self.mask
        use_cached = self.use_cached
        ipw = self.ipw

        # Create experiment directory
        if os.path.exists(output_dir) and use_cached:
            # logger.info(f"Skipping. Cached counting JTT results at {output_dir}")
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

        # Coalesce transitions a->b and b->a together
        n_time_buckets = cmats.shape[0]
        for i in range(n_time_buckets):
            cmats[i] = (cmats[i] + np.transpose(cmats[i])) / 2.0
        # Apply masking
        for i in range(n_time_buckets):
            cmats[i] = cmats[i] * mask_mat

        ##### Compute CTPs #####
        # Compute total frequency matrix (ignoring branch lengths)
        F = cmats.sum(axis=0)
        # Zero the diagonal such that summing over rows will produce the number of transitions from each state.
        F_off = F * (1.0 - np.eye(n_states))
        # Compute CTPs
        CTPs = F_off / (F_off.sum(axis=1)[:, None] + 1e-16)

        ###### Compute mutabilities #####
        if ipw:
            M = np.zeros(shape=(n_states))
            for i in range(n_time_buckets):
                qtime = qtimes[i]
                cmat = cmats[i]
                cmat_off = cmat * (1.0 - np.eye(n_states))
                M += 1.0 / qtime * cmat_off.sum(axis=1)
            M /= (F.sum(axis=1) + 1e-16)
        else:
            M = 1.0 / np.median(qtimes) * F_off.sum(axis=1) / (F.sum(axis=1) + 1e-16)

        ##### JTT estimator #####
        res = np.diag(M) @ CTPs
        np.fill_diagonal(res, -M)

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
