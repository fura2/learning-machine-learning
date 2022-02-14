'''Naive training'''

import operator
from typing import Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train(
        dataloader: DataLoader,
        model: nn.Module,
        loss_function: nn.Module,
        optimizer: Optimizer,
        device: str,
) -> None:
    '''
    Train an epoch.
    Compute accuracy and average loss for the given dataset.
    '''
    model.train()

    n_correct = 0
    total_loss = 0
    for X, y in dataloader:
        X: Tensor
        y: Tensor
        X = X.to(device)  # [batch_size, 1, 28, 28]
        y = y.to(device)  # [batch_size]

        pred: Tensor = model(X)  # [batch_size, 10]
        loss: Tensor = loss_function(pred, y)  # []

        y_pred = pred.max(dim=1).indices
        n_correct += y.eq(y_pred).sum().item()
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return n_correct / len(dataloader.dataset), total_loss / len(dataloader)


def test(
        dataloader: DataLoader,
        model: nn.Module,
        loss_function: nn.Module,
        device: str,
) -> Tuple[float, float]:
    '''Compute accuracy and average loss for the given dataset'''
    model.eval()

    n_correct = 0
    total_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X: Tensor
            y: Tensor
            X = X.to(device)  # [batch_size, 1, 28, 28]
            y = y.to(device)  # [batch_size]

            pred: Tensor = model(X)  # [batch_size, 10]
            loss: Tensor = loss_function(pred, y)  # []

            y_pred = pred.max(dim=1).indices
            n_correct += y.eq(y_pred).sum().item()
            total_loss += loss.item()

    return n_correct / len(dataloader.dataset), total_loss / len(dataloader)
