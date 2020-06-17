# type: ignore

from typing import List, Tuple

import numpy as np  # type: ignore
import torch
import torch.nn as nn


class CDLayer(nn.Module):
    def __init__(
        self,
        weight_regularizer: float = 1e-6,
        dropout_regularizer: float = 1e-5,
        init_min: float = 0.1,
        init_max: float = 0.1,
    ):
        super(CDLayer, self).__init__()

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        init_mn = np.log(init_min) - np.log(1.0 - init_min)
        init_mx = np.log(init_max) - np.log(1.0 - init_max)

        self.p_logit = nn.Parameter(  # type: ignore
            torch.empty(1).uniform_(init_mn, init_mx)
        )

    def forward(  # type: ignore
        self, x: torch.Tensor, layer: nn.Parameter
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        p = torch.sigmoid(self.p_logit)

        out = layer(self._concrete_dropout(x, p))

        sum_of_square = 0.0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))

        weight_regularizer = (
            self.weight_regularizer * sum_of_square / (1.0 - p)  # type: ignore
        )

        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1.0 - p) * torch.log(1.0 - p)  # type: ignore

        input_dimensionality = x.size(1)  # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality

        regularization = weight_regularizer + dropout_regularizer
        return out, regularization

    def _concrete_dropout(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        temp = 0.1

        unif_noise = torch.rand_like(x)

        drop_prob = (
            torch.log(p + eps)
            - torch.log(1 - p + eps)
            + torch.log(unif_noise + eps)
            - torch.log(1 - unif_noise + eps)
        )

        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p

        x = torch.mul(x, random_tensor)
        x /= retain_prob

        return x

    def p_float(self) -> float:
        return torch.sigmoid(self.p_logit).item()

    def p(self) -> str:
        return f"{torch.sigmoid(self.p_logit).item():.4f}"


class ConcreteDropoutUCIHomoscedastic(nn.Module):
    def __init__(self, ft: int, h_dim: int, wr: float = 1e-6, dr: float = 1e-5):
        super(ConcreteDropoutUCIHomoscedastic, self).__init__()

        self.h_dim = h_dim

        self.linear1 = nn.Linear(ft, h_dim)
        self.linear2 = nn.Linear(h_dim, h_dim)

        self.linear_mu = nn.Linear(h_dim, 1)

        self.conc_drop1 = CDLayer(wr, dr)
        self.conc_drop2 = CDLayer(wr, dr)

        self.conc_drop_mu = CDLayer(wr, dr)

        self.relu = nn.ReLU()

    def gaussian_mixture(
        self, x: torch.Tensor, samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mus = torch.zeros(samples, x.size(0), device=x.device)
        logvars = torch.zeros(samples, x.size(0), device=x.device)
        for i in range(samples):
            x1, _ = self.conc_drop1(x, nn.Sequential(self.linear1, self.relu))
            x2, _ = self.conc_drop1(x1, nn.Sequential(self.linear2, self.relu))
            mu, _ = self.conc_drop_mu(x2, self.linear_mu)

            mus[i] = mu.squeeze(1)

        mu = mus.mean(dim=0)
        var = ((torch.exp(logvars)) + (mus ** 2)).mean(dim=0) - (mu ** 2)
        return (mu, var)

    def forward(self, x):
        regularization = torch.empty(3, device=x.device)

        x, regularization[0] = self.conc_drop1(
            x, nn.Sequential(self.linear1, self.relu)
        )
        x, regularization[1] = self.conc_drop2(
            x, nn.Sequential(self.linear2, self.relu)
        )
        mu, regularization[2] = self.conc_drop_mu(x, self.linear_mu)

        return (mu.squeeze(), regularization.sum())

    def get_p_floats(self) -> List[float]:
        return [
            self.conc_drop1.p_float(),
            self.conc_drop2.p_float(),
            self.conc_drop_mu.p_float(),
        ]

    def get_p(self) -> str:
        return (
            f"conc1: {self.conc_drop1.p()} conc2: {self.conc_drop2.p()} "
            f"conc mu: {self.conc_drop_mu.p()}"
        )


class ConcreteDropoutUCI(nn.Module):
    def __init__(self, ft: int, h_dim: int, wr: float = 1e-6, dr: float = 1e-5):
        super(ConcreteDropoutUCI, self).__init__()

        self.h_dim = h_dim

        self.linear1 = nn.Linear(ft, h_dim)
        self.linear2 = nn.Linear(h_dim, h_dim)

        self.linear_mu = nn.Linear(h_dim, 1)
        self.linear_logvar = nn.Linear(h_dim, 1)

        self.conc_drop1 = CDLayer(wr, dr)
        self.conc_drop2 = CDLayer(wr, dr)

        self.conc_drop_mu = CDLayer(wr, dr)
        self.conc_drop_logvar = CDLayer(wr, dr)

        self.relu = nn.ReLU()

    def gaussian_mixture(
        self, x: torch.Tensor, samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mus = torch.zeros(samples, x.size(0), device=x.device)
        logvars = torch.zeros(samples, x.size(0), device=x.device)
        for i in range(samples):
            x1, _ = self.conc_drop1(x, nn.Sequential(self.linear1, self.relu))
            x2, _ = self.conc_drop1(x1, nn.Sequential(self.linear2, self.relu))
            mu, _ = self.conc_drop_mu(x2, self.linear_mu)
            logvar, _ = self.conc_drop_logvar(x2, self.linear_logvar)

            mus[i] = mu.squeeze(1)
            logvars[i] = logvar.squeeze(1)

        mu = mus.mean(dim=0)
        var = ((torch.exp(logvars)) + (mus ** 2)).mean(dim=0) - (mu ** 2)
        return (mu, var)

    def forward(self, x):
        regularization = torch.empty(4, device=x.device)

        x, regularization[0] = self.conc_drop1(
            x, nn.Sequential(self.linear1, self.relu)
        )
        x, regularization[1] = self.conc_drop2(
            x, nn.Sequential(self.linear2, self.relu)
        )

        mu, regularization[2] = self.conc_drop_mu(x, self.linear_mu)
        logvar, regularization[3] = self.conc_drop_logvar(x, self.linear_logvar)

        return (mu.squeeze(), logvar.squeeze(), regularization.sum())

    def get_p_floats(self) -> List[float]:
        return [
            self.conc_drop1.p_float(),
            self.conc_drop2.p_float(),
            self.conc_drop_mu.p_float(),
            self.conc_drop_logvar.p_float(),
        ]

    def get_p(self) -> str:
        return (
            f"conc1: {self.conc_drop1.p()} conc2: {self.conc_drop2.p()} "
            f"conc mu: {self.conc_drop_mu.p()} conc logvar: {self.conc_drop_logvar.p()}"
        )


class ConcreteDropoutMNIST(nn.Module):
    def __init__(self, ft: int, h_dim: int, wr: float = 1e-6, dr: float = 1e-5):
        """this is hte MNIST model according to the paper whihc only has 3 fully connected layers"""
        super(ConcreteDropoutMNIST, self).__init__()

        self.h_dim = h_dim

        self.linear1 = nn.Linear(ft, h_dim)
        self.linear2 = nn.Linear(h_dim, h_dim)
        self.linear3 = nn.Linear(h_dim, h_dim)

        self.linear_mu = nn.Linear(h_dim, 10)

        self.conc_drop1 = CDLayer(wr, dr)
        self.conc_drop2 = CDLayer(wr, dr)
        self.conc_drop3 = CDLayer(wr, dr)

        self.conc_drop_mu = CDLayer(wr, dr)

        self.relu = nn.ReLU()

    def gaussian_mixture(
        self, x: torch.Tensor, samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mus = torch.zeros(samples, x.size(0), device=x.device)
        logvars = torch.zeros(samples, x.size(0), device=x.device)
        for i in range(samples):
            x, _ = self.conc_drop1(x, nn.Sequential(self.linear1, self.relu))
            x, _ = self.conc_drop2(x, nn.Sequential(self.linear2, self.relu))
            mu, _ = self.conc_drop_mu(x, self.linear_mu)
            logvar, _ = self.conc_drop_logvar(x, self.linear_logvar)

            mus[i] = mu.squeeze(1)
            logvars[i] = logvar.squeeze(1)

        mu = mus.mean(dim=0)
        var = ((torch.exp(logvars)) + (mus ** 2)).mean(dim=0) - (mu ** 2)
        return (mu, var)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        regularization = torch.empty(4, device=x.device)

        x, regularization[0] = self.conc_drop1(
            x, nn.Sequential(self.linear1, self.relu)
        )
        x, regularization[1] = self.conc_drop2(
            x, nn.Sequential(self.linear2, self.relu)
        )
        x, regularization[2] = self.conc_drop3(
            x, nn.Sequential(self.linear3, self.relu)
        )

        out, regularization[3] = self.conc_drop_mu(x, self.linear_mu)

        # print(f"in forward: regularization: {regularization}")

        return (out, regularization.sum())

    def get_p_floats(self) -> List[float]:
        return [
            self.conc_drop1.p_float(),
            self.conc_drop2.p_float(),
            self.conc_drop3.p_float(),
            self.conc_drop_mu.p_float(),
        ]

    def get_p(self) -> str:
        return (
            f"conc1: {self.conc_drop1.p()} conc2: {self.conc_drop2.p()} "
            f"conc3: {self.conc_drop3.p()} conc mu: {self.conc_drop_mu.p()}"
        )
