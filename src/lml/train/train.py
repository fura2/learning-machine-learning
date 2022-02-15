'''Naive training'''

from typing import Tuple

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train(
        dataloader: DataLoader,
        model: nn.Module,
        loss_function: nn.Module,
        optimizer: Optimizer,
) -> Tuple[float, float]:
    '''
    Train an epoch.
    Compute average loss and accuracy for the given dataset.
    '''
    model.train()

    total_loss = 0
    n_correct = 0
    for X, y in dataloader:
        X: Tensor  # [batch_size, 1, 28, 28]
        y: Tensor  # [batch_size]

        pred: Tensor = model(X)  # [batch_size, 10]
        loss: Tensor = loss_function(pred, y)  # []

        total_loss += loss.item()
        y_pred = pred.max(dim=1).indices
        n_correct += y.eq(y_pred).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader), n_correct / len(dataloader.dataset)


def test(
        dataloader: DataLoader,
        model: nn.Module,
        loss_function: nn.Module,
) -> Tuple[float, float]:
    '''Compute average loss and accuracy for the given dataset'''
    model.eval()

    total_loss = 0
    n_correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X: Tensor  # [batch_size, 1, 28, 28]
            y: Tensor  # [batch_size]

            pred: Tensor = model(X)  # [batch_size, 10]
            loss: Tensor = loss_function(pred, y)  # []

            total_loss += loss.item()
            y_pred = pred.max(dim=1).indices
            n_correct += y.eq(y_pred).sum().item()

    return total_loss / len(dataloader), n_correct / len(dataloader.dataset)
