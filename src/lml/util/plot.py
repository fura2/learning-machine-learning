import csv
import matplotlib.pyplot as plt


def main() -> None:
    epochs = []
    accuracies = []
    avg_losses = []
    with open('output.csv', 'r') as f:
        reader = csv.reader(f)
        for i, (epoch, acc, loss) in enumerate(reader):
            if i == 0:
                continue
            epochs.append(int(epoch))
            accuracies.append(float(acc))
            avg_losses.append(float(loss))

    fig = plt.figure()
    axes1 = fig.subplots()
    axes2 = axes1.twinx()
    axes1.plot(epochs, accuracies, color='blue')
    axes2.plot(epochs, avg_losses, color='red')
    plt.show()


if __name__ == '__main__':
    main()
