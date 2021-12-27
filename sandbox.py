import csv
import operator
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from model.simple_4layers import SimpleFourLayersNetwork
from model.logistic_regression import LogisticRegression


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


def train(
        dataloader: DataLoader,
        model: nn.Module,
        loss_function: nn.Module,
        optimizer: Optimizer,
        device: str,
) -> None:
    model.train()
    for X, y in dataloader:
        X: Tensor
        y: Tensor
        X = X.to(device)  # [batch_size, 1, 28, 28]
        y = y.to(device)  # [batch_size]

        pred: Tensor = model(X)  # [batch_size, 10]
        loss: Tensor = loss_function(pred, y)  # []

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(
        dataloader: DataLoader,
        model: nn.Module,
        loss_function: nn.Module,
        device: str,
) -> Tuple[float, float]:
    model.eval()

    correct = 0
    total_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X: Tensor
            y: Tensor
            X = X.to(device)  # [batch_size, 1, 28, 28]
            y = y.to(device)  # [batch_size]

            pred: Tensor = model(X)  # [batch_size, 10]
            loss: Tensor = loss_function(pred, y)  # []

            y = y.detach().numpy()
            y_pred = np.argmax(pred.detach().numpy(), axis=1)
            correct += sum(map(operator.eq, y, y_pred))
            total_loss += loss.item()

    return correct / len(dataloader.dataset), total_loss / len(dataloader)


def review(
        dataloader: DataLoader,
        model: nn.Module,
        device: str,
        wrong_only: bool = False,
) -> None:
    model.eval()

    def get_predicted_label(X: Tensor) -> int:
        X = X.to(device)
        pred = model(X)
        return int(np.argmax(pred.detach().numpy()))

    testdata = [
        (_X, _y)
        for X, y in dataloader
        for _X, _y in zip(X, y)
    ]
    if wrong_only:
        testdata = [(X, y) for (X, y) in testdata if get_predicted_label(X) != y.item()]

    n = (len(testdata) + 63) // 64
    for t in range(n):
        fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(7, 7))
        fig.canvas.manager.set_window_title('Review')
        fig.suptitle(f'{t + 1}/{n} (enter: next page, q: quit)')

        for i, (X, y) in enumerate(testdata[64*t : 64*(t+1)]):
            y_pred = get_predicted_label(X)

            row = i // 8
            col = i % 8
            X.squeeze_()  # [1, 28, 28] -> [28, 28]
            y = y.item()
            if y == y_pred:
                axes[row, col].set_title(y, color='green')
            else:
                axes[row, col].set_title(f'{y} → {y_pred}', color='red', fontweight='bold')
            axes[row, col].xaxis.set_visible(False)
            axes[row, col].yaxis.set_visible(False)
            axes[row, col].imshow(X, cmap='plasma')
        plt.subplots_adjust(hspace=0.75)

        is_over = False
        def press_event(event):
            nonlocal is_over
            if event.key == 'q':
                is_over = True
        fig.canvas.mpl_connect('key_press_event', press_event)
        plt.rcParams['keymap.quit'] = ['q', 'enter']

        plt.show()
        if is_over:
            break


def confusion_matrix(
        dataloader: DataLoader,
        model: nn.Module,
        device: str,
) -> List[List[int]]:
    conf_mat = None
    with torch.no_grad():
        for X, y in dataloader:
            X: Tensor
            y: Tensor
            X = X.to(device)  # [batch_size, 1, 28, 28]
            y = y.to(device)  # [batch_size]

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
        device: str,
) -> None:
    conf_mat = confusion_matrix(dataloader, model, device)
    h = len(conf_mat)
    w = len(conf_mat[0])

    fig = plt.figure()
    plt.imshow(conf_mat, cmap='Blues')
    fig.canvas.manager.set_window_title('Confusion matrices')
    plt.title('Confusion matrix', size=15)
    plt.colorbar()
    plt.xticks(np.arange(w))
    plt.yticks(np.arange(h))
    plt.xlabel('Predicted', size=12)
    plt.ylabel('Actual', size=12)
    threshold = np.amax(conf_mat) / 2
    for i in range(h):
        for j in range(w):
            plt.text(
                j, i, conf_mat[i][j] if conf_mat[i][j] > 0 else '',
                horizontalalignment='center',
                verticalalignment='center',
                color='black' if conf_mat[i][j] < threshold else 'white'
            )
    plt.show()


def make_adversarial_examples(
        dataloader: DataLoader,
        model: nn.Module,
        loss_function: nn.Module,
        epsilon: float = 0.1,
) -> DataLoader:
    class Dataset_AE(Dataset):
        def __init__(self, dataloader: DataLoader) -> None:
            self.size = len(dataloader.dataset)
            self.size = 1000

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
                    X_adv = torch.clamp(X + epsilon * torch.sign(X.grad), 0.0, 1.0)

                self.images.append(X_adv)
                self.targets.append(y)

        def __getitem__(self, index: int) -> Tuple[Tensor, int]:
            return self.images[index], self.targets[index].item()

        def __len__(self) -> int:
            return self.size

    return DataLoader(
        dataset=Dataset_AE(dataloader),
        batch_size=dataloader.batch_size,
    )


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataloader, test_dataloader = get_dataloaders()

    model = LogisticRegression().to(device)
    loss_function = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3)

    # n_epochs = 500
    # accuracies = []
    # avg_losses = []
    # for t in range(n_epochs):
    #     train(train_dataloader, model, loss_function, optimizer, device)
    #     accuracy, avg_loss = test(test_dataloader, model, loss_function, device)
    #     print(f'Epoch {t+1:3d}: Accuracy = {100*accuracy:.2f}%, Avg loss = {avg_loss:.3f}')
    #     accuracies.append(accuracy)
    #     avg_losses.append(avg_loss)
    # print('Done!')

    # torch.save(model.state_dict(), 'model.pth')
    # print('Saved model to model.pth')

    # with open('output.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['Epoch', 'Accuracy', 'Avg loss'])
    #     for epoch, (acc, loss) in enumerate(zip(accuracies, avg_losses)):
    #         writer.writerow([epoch + 1, acc, loss])
    #     print('Saved accuracy data to output.csv')

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
