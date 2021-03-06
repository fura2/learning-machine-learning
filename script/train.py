import csv
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn

from lml.model import LogisticRegression, TutorialNetwork
from lml.train import test, train
from lml.util import get_dataloaders


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('n_epochs', type=int, help='number of epochs')
    parser.add_argument(
        '--output-model',
        type=Path,
        default=Path('model.pth'),
        help='path to model file (*.pth)',
    )
    parser.add_argument(
        '--output-history',
        type=Path,
        default=Path('history.csv'),
        help='path to history file (*.csv)',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_epochs = args.n_epochs
    model_path = args.output_model
    history_path = args.output_history

    train_dataloader, test_dataloader = get_dataloaders()

    # model = LogisticRegression()
    model = TutorialNetwork()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3)

    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []
    for t in range(n_epochs):
        train_loss, train_acc = train(train_dataloader, model, loss_function, optimizer)
        test_loss, test_acc = test(test_dataloader, model, loss_function)

        print(f'Epoch {t:{len(str(n_epochs-1))}d}: Loss = {test_loss:.3f}, Accuracy = {100*test_acc:.2f}%')
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
    print('Done!')

    torch.save(model.state_dict(), model_path)
    print(f'Saved model to {model_path.as_posix()}')

    with history_path.open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy'])
        writer.writerows(
            [epoch, *quad]
            for epoch, quad in enumerate(zip(
                train_losses, train_accuracies,
                test_losses, test_accuracies,
            ))
        )
        print(f'Saved history to {history_path.as_posix()}')


if __name__ == '__main__':
    main()
