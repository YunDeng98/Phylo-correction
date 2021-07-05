import torch
import torch.nn as nn


class RateMatrix(nn.Module):
    def __init__(self, num_states, mode, pi=None, pi_requires_grad=False):
        super().__init__()
        if pi is not None:
            assert pi.ndim == 1
            pi_logits = torch.log(pi)
            self._pi = nn.parameter.Parameter(pi_logits.clone(), requires_grad=pi_requires_grad)
        self.num_states = num_states
        self.mode = mode
        nparams_half = int(0.5 * num_states * (num_states - 1))
        self.upper_diag = nn.parameter.Parameter(
            0.01 * torch.randn(nparams_half, requires_grad=True)
        )
        if (mode in ["default", "stationary", "pande"]):
            self.lower_diag = nn.parameter.Parameter(
                0.01 * torch.randn(nparams_half, requires_grad=True)
            )
        self.activation = nn.Softplus()

    # @property
    # def pi(self):
    #     return nn.Softmax()(self._pi)

    # @property
    # def pi_mat(self):
    #     return torch.diag(self.pi)

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
            rmat_off = torch.zeros(self.num_states, self.num_states, device=device)
            triu_indices = torch.triu_indices(
                row=self.num_states, col=self.num_states, offset=1, device=device
            )
            rmat_off[triu_indices[0], triu_indices[1]] = self.activation(self.upper_diag)
            if self.mode == "stationary_reversible":
                rmat_off = rmat_off + rmat_off.T
            elif self.mode == "stationary":
                tril_indices = torch.tril_indices(
                    row=self.num_states, col=self.num_states, offset=-1, device=device
                )
                rmat_off[tril_indices[0], tril_indices[1]] = self.activation(
                    self.lower_diag
                )
            pi = nn.Softmax()(self._pi)
            pi_mat = torch.diag(pi)
            rmat_diag = -(rmat_off @ pi) / pi
            rmat = rmat_off + torch.diag(rmat_diag)

            mat = rmat @ pi_mat

        if self.mode == "pande_reversible":
            rmat_off = torch.zeros(self.num_states, self.num_states, device=device)
            triu_indices = torch.triu_indices(
                row=self.num_states, col=self.num_states, offset=1, device=device
            )
            rmat_off[triu_indices[0], triu_indices[1]] = self.activation(self.upper_diag)
            rmat_off = rmat_off + rmat_off.T
            
            pi = nn.Softmax(dim=-1)(self._pi)
            pi_mat = torch.diag(pi.sqrt())
            pi_inv_mat = torch.diag(pi.sqrt()**(-1))
            mat = (pi_inv_mat @ rmat_off) @ pi_mat
            mat -= torch.diag(mat.sum(1))

        if self.mode == "pande":
            rmat_off = torch.zeros(self.num_states, self.num_states, device=device)
            triu_indices = torch.triu_indices(
                row=self.num_states, col=self.num_states, offset=1, device=device
            )
            rmat_off[triu_indices[0], triu_indices[1]] = self.activation(self.upper_diag)
            tril_indices = torch.tril_indices(
                    row=self.num_states, col=self.num_states, offset=-1, device=device
                )
            rmat_off[tril_indices[0], tril_indices[1]] = self.activation(
                self.lower_diag
            )
        
            pi = nn.Softmax(-1)(self._pi)
            pi_mat = torch.diag(pi.sqrt())
            pi_inv_mat = torch.diag(pi.sqrt()**(-1))
            mat = (pi_inv_mat @ rmat_off) @ pi_mat
            mat -= torch.diag(mat.sum(1))
            # mat += torch.eye(self.num_states, device=mat.device) * 
        return mat
