import argparse
import csv

import numpy as np
import torch
from tqdm import tqdm

from concrete_dropout import ConcreteDropoutUCI  # type: ignore
from dataloader import get_pbp_sets, pbp_sets  # type: ignore
from utils import Stats, log_likelihood

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="instances in each batch")
    parser.add_argument("--samples", type=int, default=10000, help="number of test time MC samples")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--val", action="store_true", help="use a validation set or not")
    args = parser.parse_args()
    # fmt: on

    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(0)

    hypers = {
        "bostonHousing": [[0.0001, 0.1], [1.0, 1.0]],
        "concrete": [[0.0001, 1.0], [0.01, 1.0]],
        "energy": [[0.1, 1.0], [0.01, 1.0]],
        "kin8nm": [[0.001, 1.0], [0.1, 1.0]],
        "power-plant": [[2.0, 1.0], [1.0, 0.1]],
        "wine-quality": [[0.01, 0.01], [1.0, 0.1]],
        "yacht": [[0.001, 1.0], [2.0, 1.0]],
        "naval-propulsion": [[0.1, 1.0], [0.001, 1.0]],
        "protein-tertiary-structure": [[0.0001, 1.0], [2.0, 1.0]],
    }

    for name in pbp_sets:
        for (lengthscale, tau) in hypers[name]:
            stats = Stats()
            for run in range(10):
                train, val, test = get_pbp_sets(name, 32, get_val=args.val)

                # l = prior lengthscale, tau = model precision, N = dataset instances
                # weight regularizer = l^2 / (tau N)
                wr = lengthscale ** 2 / (tau * len(train.dataset))
                # dropout regularizer = 2 / tau N
                dr = 2 / (tau * len(train.dataset))

                for (x, y) in train:
                    break

                model = ConcreteDropoutUCI(x.size(1), 50, wr=wr, dr=dr).to(device)
                optimizer = torch.optim.Adam(model.parameters())
                log = tqdm(total=0, leave=False, bar_format="{desc}", position=2)

                model.train()
                its = 0

                # max = train for 40 epochs or at least 1000 batches. min = stop at 10^4 batches if 40 epochs is
                # over that much (for larger datasets)
                epochs = min(
                    10000 / (len(train.dataset) / args.batch_size),
                    max(1000 / (len(train.dataset) / args.batch_size), args.epochs),
                )
                for epochs in tqdm(range(int(epochs)), leave=False, position=0):
                    total_ll = 0.0
                    squared_err = torch.tensor(0.0, device=device)
                    for i, (x, y) in enumerate(tqdm(train, leave=False, position=1)):
                        x, y = x.to(device), y.to(device)

                        mus, logvars, regularization = model(x)

                        ll = log_likelihood(y, mus, logvars)
                        total_ll += ll.sum().item()

                        real_y = y * train.dataset.y_sigma + train.dataset.y_mu  # type: ignore
                        real_mu = mus * test.dataset.y_sigma + test.dataset.y_mu  # type: ignore
                        squared_err += ((real_y - real_mu) ** 2).sum().item()

                        loss = torch.exp(-logvars) * (y - mus) ** 2 + logvars
                        loss = loss.mean() + regularization

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    total_ll /= len(train.dataset)
                    squared_err = (squared_err / len(train.dataset)) ** 0.5

                    log.set_description(
                        f"total_ll: {total_ll:.4f} rmse: {squared_err:.4f} p: {model.get_p()}"
                    )

                model.eval()
                with torch.no_grad():
                    index = 0
                    squared_err = torch.tensor(0.0, device=device)
                    total_ll = 0.0
                    for i, (x, y) in enumerate(test):
                        x, y = x.to(device), y.to(device)

                        mus = torch.zeros(args.samples, x.size(0), device=device)
                        logvars = torch.zeros(args.samples, x.size(0), device=device)

                        for j in range(args.samples):
                            mus[j], logvars[j], _ = model(x)

                        ll = log_likelihood(y, mus, logvars)
                        ll = torch.logsumexp(ll.sum(dim=1), dim=0) - np.log(
                            args.samples
                        )
                        total_ll += ll.item()

                        index += x.size(0)

                        real_y = y * test.dataset.y_sigma + test.dataset.y_mu  # type: ignore
                        real_mu = mus.mean(dim=0) * test.dataset.y_sigma + test.dataset.y_mu  # type: ignore
                        squared_err += ((real_y - real_mu) ** 2).sum().item()

                    total_ll /= len(test.dataset)
                    squared_err = (squared_err / len(test.dataset)) ** 0.5
                    print(f"{test.dataset} ll: {total_ll:.4f} rmse: {squared_err:.4f}")
                    stats.set(total_ll, squared_err.item())
                    print(f"running stats: {stats}")

            with open("results.csv", mode="a+") as f:
                writer = csv.writer(
                    f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                writer.writerow(
                    [
                        f"{name}",
                        lengthscale,
                        tau,
                        *model.get_p_floats(),
                        stats.ll,
                        stats.rmse,
                    ]
                )
