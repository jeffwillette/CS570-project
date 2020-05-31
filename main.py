import argparse

import numpy as np  # type: ignore
import torch

from concrete_dropout import ConcreteDropoutModel  # type: ignore
from dataloader import get_pbp_sets, pbp_sets  # type: ignore

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="instances in each batch")
    parser.add_argument("--samples", type=int, default=100, help="number of test time MC samples")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    args = parser.parse_args()
    # fmt: on

    device = torch.device(f"cuda:{args.gpu}")

    for name in pbp_sets:
        train, val, test = get_pbp_sets(name, 32, get_val=False)

        for (x, y) in train:
            break

        model = ConcreteDropoutModel(x.size(1), 50).to(device)
        optimizer = torch.optim.Adam(model.parameters())

        model.train()
        for epoch in range(args.epochs):
            for i, (x, y) in enumerate(train):
                x, y = x.to(device), y.to(device)

                mu, logvar, regularization = model(x)

                loss = (y - mu) ** 2 / torch.exp(logvar) + logvar
                loss = loss.mean() + regularization

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(test):
                x, y = x.to(device), y.to(device)

                mus = torch.zeros(args.samples, x.size(0), device=device)
                logvars = torch.zeros(args.samples, x.size(0), device=device)

                for j in range(args.samples):
                    mus[j], logvars[j], _ = model(x)

                ll = (
                    -0.5 * torch.exp(-logvars) * (mus - y) ** 2
                    - 0.5 * logvars
                    - 0.5 * np.log(2 * np.pi)
                )

                # TODO: stopped here, implement the metric as they did in the oroginal repo

                print(ll.mean())
                # ll =  ll.sum(dim=1)
                # a_max = ll.max(dim=0)
                # torch.log()
