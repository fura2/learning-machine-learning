import csv

import numpy as np
import torch
from torch import Tensor, nn

from lml.train import train, test
from lml.model import LogisticRegression, TutorialNetwork
from lml.sandbox import get_dataloaders


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataloader, test_dataloader = get_dataloaders()

    model = TutorialNetwork().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3)

    n_epochs = 500
    accuracies = []
    avg_losses = []
    for t in range(n_epochs):
        train(train_dataloader, model, loss_function, optimizer, device)
        accuracy, avg_loss = test(test_dataloader, model, loss_function, device)
        print(f'Epoch {t+1:3d}: Accuracy = {100*accuracy:.2f}%, Avg loss = {avg_loss:.3f}')
        accuracies.append(accuracy)
        avg_losses.append(avg_loss)
    print('Done!')

    torch.save(model.state_dict(), 'model.pth')
    print('Saved model to model.pth')

    with open('output.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Accuracy', 'Avg loss'])
        for epoch, (acc, loss) in enumerate(zip(accuracies, avg_losses)):
            writer.writerow([epoch + 1, acc, loss])
        print('Saved accuracy data to output.csv')


if __name__ == '__main__':
    main()
