import argparse
import csv
from typing import Dict, Tuple

import numpy as np  # type: ignore
import torch
from torch import nn
from torch.distributions import Normal
from tqdm import tqdm  # type: ignore

from concrete_dropout import ConcreteDropoutUCIHomoscedastic  # type: ignore
from dataloader import get_pbp_sets  # type: ignore
from utils import Stats

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="instances in each batch")
    parser.add_argument("--samples", type=int, default=10000, help="number of test time MC samples")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--runs", type=int, default=20, help="the number of runs to do")
    args = parser.parse_args()
    # fmt: on

    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(0)

    lengthscales = [0.0001, 0.001, 0.001, 0.01, 0.1, 1.0]
    taus: Dict[str, Tuple[float, ...]] = {
        "bostonHousing": (0.1, 0.15, 0.20),
        "concrete": (0.025, 0.05, 0.075),
        "energy": (0.25, 0.5, 0.75),
        "kin8nm": (150, 200, 250),
        "power-plant": (0.05, 0.1, 0.15),
        "wine-quality-red": (2.5, 3.0, 3.5),
        "yacht": (0.25, 0.5, 0.75),
        "naval-propulsion-plant": (30000, 40000, 50000),
        "protein-tertiary-structure": (0.025, 0.05, 0.075),
    }

    for name in taus:
        print(name)
        best_rmse = float("inf")
        best_stats = None
        best_model = None
        best_hypers = (0.0, 0.0)

        for lengthscale in lengthscales:
            for tau in taus[name]:
                stats = Stats()
                for run in range(args.runs):
                    train, _, test = get_pbp_sets(name, args.batch_size, get_val=False)

                    # l = prior lengthscale, tau = model precision, N = dataset instances
                    # weight regularizer = l^2 / (tau N)
                    wr = lengthscale ** 2.0 / (tau * len(train.dataset))
                    # dropout regularizer = 2 / tau N
                    dr = 2 / (tau * len(train.dataset))

                    for (x, y) in train:
                        break

                    h_dim = 50 if "protein" not in name else 100
                    model = ConcreteDropoutUCIHomoscedastic(
                        x.size(1), h_dim, wr=wr, dr=dr
                    ).to(device)
                    optimizer = torch.optim.Adam(model.parameters())
                    log = tqdm(total=0, leave=False, bar_format="{desc}", position=2)

                    best_model = model.state_dict()
                    # epochs = max(1000 / (len(train.dataset) / args.batch_size), 40)
                    # epochs = min(10000 / (len(train.dataset) / args.batch_size), epochs)
                    for epoch in tqdm(
                        range(int(args.epochs * 10)), leave=False, position=0
                    ):
                        model.train()
                        for i, (x, y) in enumerate(
                            tqdm(train, leave=False, position=1)
                        ):
                            x, y = x.to(device), y.to(device)

                            mu, regularization = model(x)

                            loss = ((y - mu) ** 2).sum() + regularization

                            optimizer.zero_grad()
                            loss.backward(retain_graph=True)

                            optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        squared_err = 0.0
                        total_ll = 0.0
                        aleatoric = 1 / tau
                        epistemic = 0.0
                        # total_d_ll = 0.0
                        for i, (x, y) in enumerate(test):
                            x, y = x.to(device), y.to(device)

                            mus = torch.zeros(args.samples, x.size(0), device=device)
                            for j in range(args.samples):
                                mus[j], _ = model(x)

                            nat_mu = mus * train.dataset.y_sigma + train.dataset.y_mu  # type: ignore
                            nat_y = y * train.dataset.y_sigma + train.dataset.y_mu  # type: ignore

                            # this works with the tau outside of the logsumexp because tau is constant for
                            # all predictions. tau is also 1/sigma^2 so it it log(1) - log(sigma^2) which turns
                            # it into the plain equation for negative log likelihood.
                            total_ll += (
                                (
                                    torch.logsumexp(
                                        -0.5 * tau * (nat_y - nat_mu) ** 2, dim=0
                                    ).cpu()
                                    - np.log(args.samples)
                                    - 0.5 * np.log(2 * np.pi)
                                    + 0.5 * np.log(tau)
                                )
                                .sum()
                                .item()
                            )

                            epistemic += mus.var(dim=0).mean().item()
                            # aleatoric += torch.exp(logvar).mean(dim=0).sum().item()

                            real_y = y * test.dataset.y_sigma + test.dataset.y_mu  # type: ignore
                            real_mu = mus.mean(dim=0) * test.dataset.y_sigma + test.dataset.y_mu  # type: ignore
                            squared_err += ((real_y - real_mu) ** 2).sum().item()

                        aleatoric /= len(test.dataset)
                        epistemic /= len(test.dataset)

                        total_ll /= len(test.dataset)

                        squared_err = (squared_err / len(test.dataset)) ** 0.5
                        stats.set(total_ll, squared_err, aleatoric, epistemic)

                        print(
                            f"{test.dataset} ll: {total_ll:.4f} rmse: {squared_err:.4f} running: {stats}"
                        )

                        with open(
                            f"results-homo/{test.dataset}-batch-{args.batch_size}-lengthscale-{lengthscale}-tau-{tau}-runs.csv",
                            mode="a+",
                        ) as f:
                            writer = csv.writer(
                                f,
                                delimiter=",",
                                quotechar='"',
                                quoting=csv.QUOTE_MINIMAL,
                            )
                            writer.writerow(
                                [
                                    run,
                                    *model.get_p_floats(),
                                    total_ll,
                                    squared_err,
                                    aleatoric,
                                    epistemic,
                                ]
                            )
                if stats.get_rmse()[0] < best_rmse:
                    best_rmse = stats.get_rmse()[0]
                    best_stats = stats
                    best_model = model.state_dict()
                    best_hypers = (lengthscale, tau)

        # save model to file, and load these parameters into the model
        torch.save(
            model.state_dict(),
            f"trained-homo/{train.dataset}-{lengthscale}-{tau}-run-{run}.pt",
        )

        model.load_state_dict(best_model)
        if best_stats is None:
            raise ValueError("best stats cannot be none here")

        with open(
            f"results-homo/uci-results-batch-{args.batch_size}.csv", mode="a+",
        ) as f:
            writer = csv.writer(
                f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            writer.writerow(
                [
                    f"{name}",
                    best_hypers[0],
                    best_hypers[1],
                    *model.get_p_floats(),
                    *best_stats.get_ll(),
                    *best_stats.get_rmse(),
                    *best_stats.get_uncertainties(),
                ]
            )
