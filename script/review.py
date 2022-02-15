from argparse import ArgumentParser
from pathlib import Path

import torch

from lml.model import LogisticRegression, TutorialNetwork
from lml.util import get_dataloaders, review


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('model_path', type=Path, help='path to model file (*.pth)')
    parser.add_argument('--wrong-only', action='store_true', help='review incorrect instances only')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path
    assert model_path.exists(), f'{model_path.as_posix()} does not exists'

    _, test_dataloader = get_dataloaders()

    model = TutorialNetwork()
    model.load_state_dict(torch.load(model_path))
    review(test_dataloader, model, wrong_only=args.wrong_only)


if __name__ == '__main__':
    main()
