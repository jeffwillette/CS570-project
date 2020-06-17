import argparse
import csv
from typing import Dict, Tuple

import numpy as np  # type: ignore
import torch
from torch.distributions import Normal
from tqdm import tqdm  # type: ignore

from concrete_dropout import ConcreteDropoutUCI  # type: ignore
from dataloader import get_pbp_sets  # type: ignore
from utils import Stats

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="instances in each batch")
    parser.add_argument("--samples", type=int, default=10000, help="number of test time MC samples")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--val", action="store_true", help="use a validation set or not")
    parser.add_argument("--runs", type=int, default=20, help="the number of runs to do")
    args = parser.parse_args()
    # fmt: on

    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(0)

    hypers: Dict[str, Tuple[float, float]] = {
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

    for name in hypers:
        print(name)
        (lengthscale, tau) = hypers[name]
        stats = Stats()
        for run in range(args.runs):
            train, val, test = get_pbp_sets(name, args.batch_size, get_val=args.val)

            # l = prior lengthscale, tau = model precision, N = dataset instances
            # weight regularizer = l^2 / (tau N)
            wr = lengthscale ** 2.0 / (tau * len(train.dataset))
            # dropout regularizer = 2 / tau N
            dr = 2 / (tau * len(train.dataset))

            for (x, y) in train:
                break

            h_dim = 50 if "protein" not in name else 100
            model = ConcreteDropoutUCI(x.size(1), h_dim, wr=wr, dr=dr).to(device)
            optimizer = torch.optim.Adam(model.parameters())
            log = tqdm(total=0, leave=False, bar_format="{desc}", position=2)

            best_model = model.state_dict()

            epochs = max(1000 / (len(train.dataset) / args.batch_size), 40)
            epochs = min(10000 / (len(train.dataset) / args.batch_size), epochs)
            for epoch in tqdm(range(int(epochs)), leave=False, position=0):
                model.train()
                for i, (x, y) in enumerate(tqdm(train, leave=False, position=1)):
                    x, y = x.to(device), y.to(device)

                    mus, logvars, regularization = model(x)

                    loss = torch.exp(-logvars) * (y - mus) ** 2 + logvars
                    loss = loss.sum() + regularization

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # save model to file, and load these parameters into the model
            torch.save(
                model.state_dict(),
                f"trained/{train.dataset}-{lengthscale}-{tau}-run-{run}.pt",
            )

            model.eval()
            with torch.no_grad():
                squared_err = 0.0
                total_ll = 0.0
                aleatoric = 0.0
                epistemic = 0.0
                # total_d_ll = 0.0
                for i, (x, y) in enumerate(test):
                    x, y = x.to(device), y.to(device)

                    mus = torch.zeros(args.samples, x.size(0), device=device)
                    logvars = torch.zeros(args.samples, x.size(0), device=device)
                    for j in range(args.samples):
                        mus[j], logvars[j], _ = model(x)

                    ll = Normal(mus, torch.exp(logvars / 2)).log_prob(y)
                    total_ll += torch.logsumexp(ll.sum(dim=1), dim=0).item()

                    # mean = mus.mean(dim=0)
                    # std = (
                    #     mus.var(dim=0) + torch.exp(logvars / 2).mean(dim=0) ** 2
                    # ) ** 0.5
                    # total_d_ll += Normal(mean, std).log_prob(y).sum()

                    epistemic += mus.var(dim=0).mean().item()
                    aleatoric += torch.exp(logvars).mean(dim=0).sum().item()

                    real_y = y * test.dataset.y_sigma + test.dataset.y_mu  # type: ignore
                    real_mu = mus.mean(dim=0) * test.dataset.y_sigma + test.dataset.y_mu  # type: ignore
                    squared_err += ((real_y - real_mu) ** 2).sum().item()

                aleatoric /= len(test.dataset)
                epistemic /= len(test.dataset)

                total_ll = total_ll - np.log(args.samples)
                total_ll /= len(test.dataset)
                total_ll -= np.log(test.dataset.y_sigma.item())  # type: ignore

                # total_d_ll /= len(test.dataset)
                # total_d_ll -= np.log(test.dataset.y_sigma.item())  # type: ignore
                # print(f"total_ll: {total_ll} total_d_ll: {total_d_ll}")

                squared_err = (squared_err / len(test.dataset)) ** 0.5

                stats.set(total_ll, squared_err, aleatoric, epistemic)

                print(f"{test.dataset} ll: {total_ll:.4f} rmse: {squared_err:.4f}")
                print(f"running stats: {stats}")

                with open(
                    f"results/{test.dataset}-batch-{args.batch_size}-lengthscale-{lengthscale}-tau-{tau}-runs.csv",
                    mode="a+",
                ) as f:
                    writer = csv.writer(
                        f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL,
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

        with open(f"results/uci-results-batch-{args.batch_size}.csv", mode="a+",) as f:
            writer = csv.writer(
                f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            writer.writerow(
                [
                    f"{name}",
                    lengthscale,
                    tau,
                    *model.get_p_floats(),
                    *stats.get_ll(),
                    *stats.get_rmse(),
                    *stats.get_uncertainties(),
                ]
            )
