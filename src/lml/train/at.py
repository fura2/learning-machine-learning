'''Adversarial training (https://arxiv.org/abs/1412.6572)'''
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


def make_adversarial_examples(
        dataloader: DataLoader,
        model: nn.Module,
        loss_function: nn.Module,
        epsilon: float = 0.1,
) -> DataLoader:
    '''Make adversarial examples by the fast gradient sign method'''
    class DatasetAE(Dataset):
        def __init__(self, dataloader: DataLoader) -> None:
            self.size = len(dataloader.dataset)

            self.images = []
            self.targets = []
            for index in range(self.size):
                X, y = dataloader.dataset[index]
                X.requires_grad_()  # [1, 28, 28]
                assert X.grad is None
                y = torch.tensor(y)  # []

                X2 = X.unsqueeze(0)  # [1, 1, 28, 28]
                y2 = y.unsqueeze(0)  # [1]
                pred = model(X2)  # [1, 10]
                loss = loss_function(pred, y2)  # []

                loss.backward()
                with torch.no_grad():
                    X_adv = torch.clamp(X + epsilon*torch.sign(X.grad), 0.0, 1.0)

                self.images.append(X_adv)
                self.targets.append(y)

        def __getitem__(self, index: int) -> Tuple[Tensor, int]:
            return self.images[index], self.targets[index].item()

        def __len__(self) -> int:
            return self.size

    return DataLoader(
        dataset=DatasetAE(dataloader),
        batch_size=dataloader.batch_size,
    )
