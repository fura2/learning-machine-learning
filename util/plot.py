import csv
import matplotlib.pyplot as plt


def main():
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
    ax1 = fig.subplots()
    ax2 = ax1.twinx()
    ax1.plot(epochs, accuracies, color='blue')
    ax2.plot(epochs, avg_losses, color='red')
    plt.show()


if __name__ == '__main__':
    main()
