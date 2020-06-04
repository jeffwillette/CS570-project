import numpy as np  # type: ignore
import torch


class Stats:
    def __init__(self) -> None:
        self.ll = 0.0
        self.rmse = 0.0
        self.n = 1

    def set(self, ll: float, rmse: float) -> None:
        self.ll = self.ll + 1.0 / self.n * (ll - self.ll)
        self.rmse = self.rmse + 1.0 / self.n * (rmse - self.rmse)
        self.n += 1

    def __str__(self) -> str:
        return f"ll: {self.ll} rmse: {self.rmse}"


def log_likelihood(
    y: torch.Tensor, mus: torch.Tensor, logvars: torch.Tensor
) -> torch.Tensor:
    return (  # type: ignore
        -0.5 * torch.exp(-logvars) * (mus - y) ** 2
        - 0.5 * logvars
        - 0.5 * np.log(2 * np.pi)
    )
