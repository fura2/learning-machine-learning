from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from .model import LogisticRegression, TutorialNetwork
from .util import show_confusion_matrix


def get_dataloaders(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    training_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor(),
    )
    train_dataloader = DataLoader(
        dataset=training_data,
        batch_size=batch_size,
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
    )
    return train_dataloader, test_dataloader


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataloader, test_dataloader = get_dataloaders()

    model = TutorialNetwork().to(device)
    loss_function = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3)

    model.load_state_dict(torch.load('outputs/logistic_regression/model.pth'))
    show_confusion_matrix(test_dataloader, model, device)
    # review(test_dataloader, model, device, wrong_only=True)

    # ae_dataloader = make_adversarial_examples(
    #     test_dataloader,
    #     model,
    #     loss_function,
    # )
    # review(ae_dataloader, model, device, wrong_only=False)


if __name__ == '__main__':
    main()
