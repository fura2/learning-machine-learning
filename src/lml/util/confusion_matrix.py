import math
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor, nn
from torch.utils.data import DataLoader


def get_confusion_matrix(
        dataloader: DataLoader,
        model: nn.Module,
) -> List[List[int]]:
    '''Compute confusion matrix'''
    conf_mat = None
    with torch.no_grad():
        for X, y in dataloader:
            X: Tensor  # [batch_size, 1, 28, 28]
            y: Tensor  # [batch_size]

            pred: Tensor = model(X)  # [batch_size, 10]

            if conf_mat is None:
                assert len(pred.shape) == 2
                n = pred.shape[1]
                conf_mat = [[0] * n for _ in range(n)]

            y = y.detach().numpy()
            y_pred = np.argmax(pred.detach().numpy(), axis=1)
            for _y, _y_pred in zip(y, y_pred):
                conf_mat[_y][_y_pred] += 1

    return conf_mat


def show_confusion_matrix(
        dataloader: DataLoader,
        model: nn.Module,
) -> None:
    '''Show confusion matrix'''
    conf_mat = get_confusion_matrix(dataloader, model)
    h = len(conf_mat)
    w = len(conf_mat[0])
    assert h == w

    n_correct = sum(conf_mat[i][i] for i in range(h))
    n_total = np.sum(conf_mat)
    accuracy = n_correct / n_total

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    fig.canvas.manager.set_window_title('Confusion matrix')
    fig.suptitle(f'Confusion matrix (Accuracy={accuracy})', size=15)
    for index in range(2):
        image = axes[index].imshow(
            conf_mat, cmap='Blues',
            norm=None if index == 0 else colors.LogNorm()
        )
        axes[index].set_xticks(np.arange(w))
        axes[index].set_yticks(np.arange(h))
        axes[index].set_xlabel('Predicted', size=12)
        axes[index].set_ylabel('Actual', size=12)
        divider = make_axes_locatable(axes[index])
        cax = divider.append_axes('right', size='5%', pad=0.1)
        fig.colorbar(mappable=image, cax=cax)

        maxcnt = np.max(conf_mat)
        threshold = maxcnt / 2 if index == 0 else math.sqrt(maxcnt)
        for i in range(h):
            for j in range(w):
                axes[index].text(
                    j, i,
                    conf_mat[i][j] if conf_mat[i][j] > 0 else '',
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='black' if conf_mat[i][j] < threshold else 'white'
                )
    fig.tight_layout(w_pad=3.0)
    plt.show()
