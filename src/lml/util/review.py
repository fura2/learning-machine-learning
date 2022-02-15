import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor, nn
from torch.utils.data import DataLoader


def review(
        dataloader: DataLoader,
        model: nn.Module,
        wrong_only: bool = False,
) -> None:
    '''Review predicted results for the given dataset'''
    model.eval()

    def get_predicted_label(X: Tensor) -> int:
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
        fig.suptitle(f'Review {t + 1}/{n} (enter: next page, q: quit)')

        for i, (X, y) in enumerate(testdata[64*t: 64*(t+1)]):
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
        fig.subplots_adjust(hspace=0.75)

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
