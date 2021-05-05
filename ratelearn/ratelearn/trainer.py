import time

import pandas as pd
import torch
import torch.distributions as db
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train_quantization(
    rate_module,
    quantized_dataset,
    m=1.0,
    lr=1.0,
    num_epochs=1000,
    Q_true=None,
    optimizer=None,
):
    """
    Quantization baseline
    """
    if optimizer is None:
        optimizer = torch.optim.SGD(
            rate_module.parameters(), lr=lr, momentum=0.0, weight_decay=0
        )

    print(f"Training for {num_epochs} epochs")
    dlb = DataLoader(quantized_dataset, batch_size=1000, shuffle=False)
    start = time.time()
    df_res = pd.DataFrame()
    loss = 0.0
    rg = tqdm(range(num_epochs))
    for epoch in rg:
        optimizer.zero_grad()
        Q = rate_module()
        # Now compute the loss
        loss = 0.0
        for datapoint in dlb:
            branch_length, cmat = datapoint
            branch_length = branch_length.cuda()
            cmat = cmat.cuda()

            branch_length_ = branch_length
            mats = torch.log(torch.matrix_exp(branch_length_[:, None, None] * Q))
            mats = mats * cmat
            loss += -1 / m * mats.sum()
        # Take a gradient step.
        loss.backward(retain_graph=True)
        optimizer.step()
        rg.set_description(str(loss.item()), refresh=True)

        frob_norm = (
            torch.sqrt(torch.sum((Q - Q_true) * (Q - Q_true))).item()
            if Q_true is not None
            else 0.0
        )
        df_res = df_res.append(
            dict(
                frob_norm=frob_norm,
                loss=loss.item(),
                time=time.time() - start,
                epoch=epoch,
            ),
            ignore_index=True,
        )
    return df_res, Q


def train_quantization_N(
    rate_module,
    quantized_dataset,
    m=1.0,
    lr=1.0,
    max_iter=20,
    num_epochs=1000,
    Q_true=None,
    optimizer=None,
):
    """
    Quantization baseline
    """
    optimizer = torch.optim.LBFGS(rate_module.parameters(), lr=lr, max_iter=max_iter)

    print(f"Training for {num_epochs} epochs")
    dlb = DataLoader(quantized_dataset, batch_size=1000, shuffle=False)
    start = time.time()
    df_res = pd.DataFrame()
    loss = 0.0
    rg = tqdm(range(num_epochs))
    for epoch in rg:
        optimizer.zero_grad()
        Q = rate_module()

        def closure():
            optimizer.zero_grad()
            loss = 0.0
            for datapoint in dlb:
                branch_length, cmat = datapoint
                branch_length = branch_length.cuda()
                cmat = cmat.cuda()
                branch_length_ = branch_length
                mats = torch.log(torch.matrix_exp(branch_length_[:, None, None] * Q))
                mats = mats * cmat
                loss += -1 / m * mats.sum()
            loss.backward(retain_graph=True)
            return loss

        # Now compute the loss
        loss = 0.0
        for datapoint in dlb:
            branch_length, cmat = datapoint
            branch_length = branch_length.cuda()
            cmat = cmat.cuda()

            branch_length_ = branch_length
            mats = torch.log(torch.matrix_exp(branch_length_[:, None, None] * Q))
            mats = mats * cmat
            loss += -1 / m * mats.sum()
        # Take a gradient step.
        loss.backward(closure, retain_graph=True)
        optimizer.step()
        rg.set_description(str(loss.item()), refresh=True)

        frob_norm = (
            torch.sqrt(torch.sum((Q - Q_true) * (Q - Q_true))).item()
            if Q_true is not None
            else 0.0
        )
        df_res = df_res.append(
            dict(
                frob_norm=frob_norm,
                loss=loss.item(),
                time=time.time() - start,
                epoch=epoch,
            ),
            ignore_index=True,
        )
    return df_res, Q


@torch.no_grad()
def estimate_likelihood(
    rate_module,
    quantized_dataset,
    m=1.0,
):
    """
    Quantization baseline
    """
    dlb = DataLoader(quantized_dataset, batch_size=1000, shuffle=False)
    Q = rate_module()
    loss = 0.0
    for datapoint in dlb:
        branch_length, cmat = datapoint
        branch_length = branch_length.cuda()
        cmat = cmat.cuda()

        branch_length_ = branch_length
        mats = torch.log(torch.matrix_exp(branch_length_[:, None, None] * Q))
        mats = mats * cmat
        loss += -1 / m * mats.sum()
    return loss.item()
