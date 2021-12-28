from torch import Tensor, nn


class LogisticRegression(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28*28, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        logits = self.linear(x)
        return logits
