from typing import Tuple

import numpy as np  # type: ignore
import torch
import torch.nn as nn


class CDConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        weight_regularizer: float = 1e-6,
        dropout_regularizer: float = 1e-5,
        init_min: float = 0.1,
        init_max: float = 0.1,
    ):
        super(CDConvLayer, self).__init__()

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        init_mn = np.log(init_min) - np.log(1.0 - init_min)
        init_mx = np.log(init_max) - np.log(1.0 - init_max)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.p_logit = nn.Parameter(  # type: ignore
            torch.empty(1).uniform_(init_mn, init_mx)
        )

        self.first = True

    def get_p(self) -> float:
        return torch.sigmoid(self.p_logit).item()

    def forward(  # type: ignore
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        p = torch.sigmoid(self.p_logit)

        out = self.conv(self._concrete_dropout(x, p))

        sum_of_square = 0.0
        for name, param in self.conv.named_parameters():
            sum_of_square = sum_of_square + torch.sum(torch.pow(param, 2))  # type: ignore

        weights_regularizer = (
            self.weight_regularizer * sum_of_square / (1.0 - p)  # type: ignore
        )

        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1.0 - p) * torch.log(1.0 - p)  # type: ignore

        if self.first:
            print(f"in forward: channels should be the first index: {x.size()}")
            self.first = False

        input_dimensionality = x.size(1)  # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality

        regularization = weights_regularizer + dropout_regularizer
        return out, regularization

    def _concrete_dropout(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        temp = 2.0 / 3.0

        noise_shape = (x.size(0), x.size(1), 1, 1)
        unif_noise = torch.rand(noise_shape, device=x.device)

        # print(f"unif noise: {unif_noise.shape}")

        drop_prob = (
            torch.log(p + eps)
            - torch.log(1 - p + eps)
            + torch.log(unif_noise + eps)
            - torch.log(1 - unif_noise + eps)
        )

        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1.0 - drop_prob
        retain_prob = 1.0 - p

        x = torch.mul(x, random_tensor)
        x /= retain_prob

        return x


# TODO: this needs to be updated if it is going to be used
class ConvConcreteDropoutModel(nn.Module):
    def __init__(self, ft: int, h_dim: int, wr: float = 1e-6, dr: float = 1e-5):
        super(ConvConcreteDropoutModel, self).__init__()

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        regularization = torch.empty(4, device=x.device)

        x1, regularization[0] = self.conc_drop1(
            x, nn.Sequential(self.linear1, self.relu)
        )
        x2, regularization[1] = self.conc_drop2(
            x1, nn.Sequential(self.linear2, self.relu)
        )

        mu, regularization[2] = self.conc_drop_mu(x2, self.linear_mu)
        logvar, regularization[3] = self.conc_drop_logvar(x2, self.linear_logvar)

        return (mu.squeeze(), logvar.squeeze(), regularization.sum())
