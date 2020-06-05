import argparse
import csv

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST  # type: ignore
from torchvision.transforms import Compose, ToTensor  # type: ignore
from tqdm import tqdm  # type: ignore

from concrete_dropout import ConcreteDropoutMNIST  # type: ignore
from utils import ClassStats

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="instances in each batch")
    parser.add_argument("--samples", type=int, default=100, help="number of test time MC samples")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--runs", type=int, default=10, help="the number of runs to average")
    parser.add_argument("--val", action="store_true", help="use a validation set or not")
    args = parser.parse_args()
    # fmt: on

    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(0)

    tx = Compose([ToTensor()])

    lengthscale = 1e-2
    tau = 1.0

    stats = ClassStats()
    for run in range(args.runs):

        train_set = MNIST(
            "/home/jeff/datasets", download=True, train=True, transform=tx
        )
        test_set = MNIST(
            "/home/jeff/datasets", download=True, train=False, transform=tx
        )

        train = DataLoader(
            train_set, shuffle=True, batch_size=args.batch_size, num_workers=4
        )
        test = DataLoader(
            test_set, shuffle=False, batch_size=args.batch_size, num_workers=4
        )

        # l = prior lengthscale, tau = model precision, N = dataset instances
        # weight regularizer = l^2 / (tau N)
        wr = lengthscale ** 2 / (tau * len(train.dataset))
        # dropout regularizer = 2 / tau N
        dr = 2 / (tau * len(train.dataset))

        for (x, y) in train:
            break

        model = ConcreteDropoutMNIST(28 * 28, 512, wr=wr, dr=dr).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        log = tqdm(total=0, bar_format="{desc}", position=0)

        its = 0
        model.train()
        for epoch in tqdm(range(args.epochs), leave=False, position=2):
            correct = 0
            total = 0
            for i, (x, y) in enumerate(tqdm(train, position=1, leave=False)):
                x, y = x.to(device), y.to(device)

                mu, regularization = model(x.view(x.size(0), -1))

                loss = criterion(mu, y) + regularization

                correct += (torch.argmax(mu, dim=1) == y).sum()
                total += x.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                its += 1
            log.set_description(f"acc: {float(correct) / total} p: {model.get_p()}")

        model.eval()
        with torch.no_grad():
            correct = 0
            for i, (x, y) in enumerate(tqdm(test, position=1, leave=False)):
                x, y = x.to(device), y.to(device)

                mus = torch.zeros(args.samples, x.size(0), 10, device=device)
                for j in range(args.samples):
                    mus[j], _ = model(x.view(x.size(0), -1))

                mus = mus.mean(dim=0)
                correct += (torch.argmax(mus, dim=1) == y).sum()

            stats.set(float(correct) / len(test_set))
            print(f"test: {stats}")

    with open("mnist-results.csv", mode="a+") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            ["MNIST", lengthscale, tau, *model.get_p_floats(), *stats.get_acc()]
        )
