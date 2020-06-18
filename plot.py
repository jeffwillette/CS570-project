import csv
from typing import Dict, Tuple

import matplotlib as mpl
import numpy as np  # type: ignore
from matplotlib import pyplot as plt  # type: ignore

dataset_best: Dict[str, Tuple[float, float]] = {
    "bostonHousing": (0.0001, 0.1),
    "concrete": (0.001, 0.1),
    "energy": (0.0001, 0.1),
    "kin8nm": (0.1, 0.1),
    "power-plant": (0.1, 1.0),
    "wine-quality-red": (0.001, 0.01),
    "yacht": (0.0001, 1.0),
    "naval-propulsion-plant": (0.001, 1.0),
    "protein-tertiary-structure": (0.0001, 0.1),
}

mpl.style.use("seaborn-dark-palette")


def random_color() -> np.array:
    return np.random.rand(3)


def plot_mnist() -> None:
    # make a plot of MNIST and the dropout probabilities for every run
    with open("results/mnist-runs.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        probs = []
        for row in reader:
            probs.append([float(i) for i in row[3:7]])

        probs = np.array(probs)
        colors = [random_color() for _ in range(probs.shape[1])]
        labels = [f"layer {i+1}" for i in range(probs.shape[1])]

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.grid(color="lightgray", linestyle="--", zorder=0)
        ax.set_facecolor("whitesmoke")
        ax.set_xlabel("run")
        ax.set_ylabel("$p$ value")

        for i, (c, l) in enumerate(zip(colors, labels)):
            x = np.linspace(1, probs.shape[0], probs.shape[0])
            y = probs[:, i]

            ax.plot(x, y, color=c, label=l)

        ax.legend()
        ax.set_title("MNIST converged $p$ values")
        fig.savefig("charts/mnist-p.png")
        plt.clf()


def plot_rl_p(p: np.array) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(21, 6))

    ax1.plot(p[:, 0], label="layer 0")
    ax1.legend()
    ax2.plot(p[:, 1], label="layer 1")
    ax2.legend()
    ax3.plot(p[:, 2], label="layer 2")
    ax3.legend()

    fig.savefig("reinforcement-learning-p.png")


def plot_uci() -> None:
    # for each of the UCI datasets, plot dropout probabilities for each run
    for dataset in dataset_best:
        (lengthscale, tau) = dataset_best[dataset]

        probs = []
        metrics = []
        uncertainties = []

        with open(
            f"results/{dataset}-batch-32-lengthscale-{lengthscale}-tau-{tau}-runs.csv",
            "r",
        ) as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                probs.append([float(i) for i in row[1:5]])
                metrics.append([float(i) for i in row[5:7]])
                uncertainties.append([float(i) for i in row[7:]])

            probs = np.array(probs)
            metrics = np.array(metrics)
            uncertainties = np.array(uncertainties)

            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21, 6))
            for ax in axes:
                ax.grid(color="lightgray", linestyle="--", zorder=0)
                ax.set_facecolor("whitesmoke")
                ax.set_xlabel("run")

            (ax1, ax2, ax3) = axes

            colors = [random_color() for _ in range(probs.shape[1])]
            labels = ["layer 1", "layer 2", "mu layer", "logvar layer"]
            for i, (c, l) in enumerate(zip(colors, labels)):
                x = np.linspace(1, probs.shape[0], probs.shape[0])
                y = probs[:, i]
                ax1.plot(x, y, color=c, label=l)
            ax1.legend()
            ax1.set_title(f"{dataset} $p$ values")

            labels = ["log likelihood", "RMSE"]
            for i in range(metrics.shape[1]):
                x = np.linspace(1, probs.shape[0], probs.shape[0])
                y = metrics[:, i]
                c = colors[i]
                l = labels[i]
                ax2.plot(x, y, color=c, label=l)
            ax2.legend()
            ax2.set_title(f"{dataset} $\log p(y | x)$ and RMSE")

            labels = ["aleatoric", "epistemic"]
            for i in range(uncertainties.shape[1]):
                x = np.linspace(1, probs.shape[0], probs.shape[0])
                y = uncertainties[:, i]
                c = colors[i]
                l = labels[i]
                ax3.plot(x, y, color=c, label=l)
            ax3.legend()
            ax3.set_title(f"{dataset} aleatoric and epistemic uncertainty")

            fig.savefig(f"charts/{dataset}.png")


if __name__ == "__main__":
    # plot_mnist()
    plot_uci()
