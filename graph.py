import argparse
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('-o', '--output', type=str, help='output file name (png)')
    return parser.parse_args()

def main():
    args = parse_args()

    csv_data = pd.read_csv(args.input)

    epoch = csv_data['Epoch']
    accuracy = csv_data['Accuracy']
    avg_loss = csv_data['Avg loss']

    fig = plt.figure()
    ax1 = fig.subplots()
    ax2 = ax1.twinx()
    ax1.plot(epoch, accuracy, color='blue', label='Accuracy')
    ax2.plot(epoch, avg_loss, color='red', label='Avg loss')
    ax1.set_yticks([0.1 * i for i in range(0, 11)])
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2)

    if args.output is not None:
        output_name = args.output
        if output_name[-4:] != '.png':
            output_name += '.png'
        plt.savefig(output_name, format='png')

    plt.show()


if __name__ == '__main__':
    main()
