import csv
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('history_path', type=Path, help='path to history file (*.csv)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history_path = args.history_path
    assert history_path.exists(), f'{history_path.as_posix()} does not exists'

    epochs = []
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    with history_path.open('r') as f:
        reader = csv.reader(f)
        for t, (epoch, train_acc, train_loss, test_acc, test_loss) in enumerate(reader):
            if t == 0:
                continue
            epochs.append(int(epoch))
            train_losses.append(float(train_loss))
            train_accuracies.append(float(train_acc))
            test_losses.append(float(test_loss))
            test_accuracies.append(float(test_acc))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    fig.canvas.manager.set_window_title('Learning Curves')
    fig.suptitle('Learning Curves')
    axes[0].set_title('Loss')
    axes[0].plot(epochs, train_losses, color='red', label='train')
    axes[0].plot(epochs, test_losses, color='blue', label='test')
    axes[0].set_xlim(0, len(epochs))
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend()
    axes[1].set_title('Accuracy')
    axes[1].plot(epochs, train_accuracies, color='red', label='train')
    axes[1].plot(epochs, test_accuracies, color='blue', label='test')
    axes[1].set_xlim(0, len(epochs))
    axes[1].set_ylim(0.8, 1.0)
    axes[1].legend()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
