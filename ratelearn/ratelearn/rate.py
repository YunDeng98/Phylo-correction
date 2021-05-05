import torch
import torch.nn as nn


class RateMatrix(nn.Module):
    def __init__(self, num_states, mode, pi=None):
        super().__init__()
        if pi is not None:
            assert pi.ndim == 1
            self.pi = nn.parameter.Parameter(pi, requires_grad=False)
        self.num_states = num_states
        self.mode = mode
        nparams_half = int(0.5 * num_states * (num_states - 1))
        self.upper_diag = nn.parameter.Parameter(
            torch.randn(nparams_half, requires_grad=True)
        )
        if (mode == "default") or (mode == "stationary"):
            self.lower_diag = nn.parameter.Parameter(
                torch.randn(nparams_half, requires_grad=True)
            )
        self.activation = nn.Softplus()

    @property
    def pi_mat(self):
        return torch.diag(self.pi)

    def forward(self):
        device = self.upper_diag.device
        if self.mode == "default":
            mat = torch.zeros(
                self.num_states,
                self.num_states,
                device=device,
            )
            triu_indices = torch.triu_indices(
                row=self.num_states,
                col=self.num_states,
                offset=1,
                device=device,
            )
            mat[triu_indices[0], triu_indices[1]] = self.activation(self.upper_diag)
            tril_indices = torch.tril_indices(
                row=self.num_states, col=self.num_states, offset=-1, device=device
            )
            mat[tril_indices[0], tril_indices[1]] = self.activation(self.lower_diag)
            mat = mat - torch.diag(mat.sum(1))

        if self.mode in ["stationary_reversible", "stationary"]:
            rmat = torch.zeros(self.num_states, self.num_states, device=device)
            triu_indices = torch.triu_indices(
                row=self.num_states, col=self.num_states, offset=1, device=device
            )
            rmat[triu_indices[0], triu_indices[1]] = self.activation(self.upper_diag)
            if self.mode == "stationary_reversible":
                rmat = rmat + rmat.T
            elif self.mode == "stationary":
                tril_indices = torch.tril_indices(
                    row=self.num_states, col=self.num_states, offset=-1, device=device
                )
                rmat[tril_indices[0], tril_indices[1]] = self.activation(
                    self.lower_diag
                )
            rmat_diag = -(rmat @ self.pi) / self.pi
            rmat += torch.diag(rmat_diag)

            mat = rmat @ self.pi_mat
        return mat
