from __future__ import annotations

import io
import os
import zipfile
from typing import Any, Optional, Tuple

import numpy as np  # type: ignore
import torch
from torch.utils.data import DataLoader, Dataset
from urllib3 import PoolManager  # type: ignore

pbp_sets = [
    "bostonHousing",
    "concrete",
    "energy",
    "kin8nm",
    "power-plant",
    "wine-quality-red",
    "yacht",
    "naval-propulsion-plant",
    "protein-tertiary-structure",
]


class Loader(DataLoader):
    def __init__(self, dataset: RegressionDataset, **kwargs: Any) -> None:
        super(Loader, self).__init__(dataset, **kwargs)


class RegressionDataset(Dataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(RegressionDataset, self).__init__()
        self.x: torch.Tensor
        self.y: torch.Tensor
        self.og_y: torch.Tensor

        # these are the parameters of the normalization
        self.mu: torch.Tensor
        self.sigma: torch.Tensor
        self.y_mu: torch.Tensor
        self.y_sigma: torch.Tensor

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]

    def __len__(self) -> int:
        return self.x.size(0)

    def prune(self, idx: torch.Tensor) -> None:
        self.x = self.x[idx]
        self.y = self.y[idx]

    def set_name(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name

    def get_feature_ranges(self) -> torch.Tensor:
        """
        get the feature ranges for all of the x features, this was originally used
        to determine the ranges of features for generating adversarial examples as done by
        deep ensembles paper https://arxiv.org/abs/1612.01474
        """
        return torch.abs(self.x.min(dim=0)[0] - self.x.max(dim=0)[0])

    def valid(self) -> None:
        if torch.any(torch.isinf(self.x)) or torch.any(torch.isnan(self.x)):
            raise ValueError("x has invalid values")
        elif torch.any(torch.isinf(self.y)) or torch.any(torch.isnan(self.y)):
            raise ValueError("y has invalid values")

    def standard_normalize(
        self,
        mu: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
        y_mu: Optional[torch.Tensor] = None,
        y_sigma: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """standard normalize the dataset by ([x,y] - mu) / sigma"""
        if mu is None or sigma is None or y_mu is None or y_sigma is None:

            self.mu = self.x.mean(dim=0)
            self.sigma = self.x.std(dim=0)
            self.sigma[self.sigma == 0] = 1

            if torch.any(self.sigma == 0):
                # if sigma is zero that means mu must all be the same value, set to 1 to avoid dividing by zero
                self.sigma[self.sigma == 0] = 1

            self.x = (self.x - self.mu) / self.sigma

            self.y_mu = self.y.mean()
            self.y_sigma = self.y.std()

            self.og_y = torch.clone(self.y)
            self.y = (self.y - self.y_mu) / self.y_sigma

            self.valid()
            return self.mu, self.sigma, self.y_mu, self.y_sigma

        self.mu = mu
        self.sigma = sigma
        self.y_mu = y_mu
        self.y_sigma = y_sigma

        self.x = (self.x - self.mu) / self.sigma
        self.og_y = torch.clone(self.y)
        self.y = (self.y - self.y_mu) / self.y_sigma

        self.valid()
        return mu, sigma, y_mu, y_sigma

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """get a random sample of size n of the dataset"""
        perm = torch.randperm(self.x.size(0))
        return self.x[perm[:n]], self.y[perm[:n]]


class PBPDataset(RegressionDataset):
    def __init__(
        self, *args: Any, x: torch.Tensor = None, y: torch.Tensor = None, name: str = ""
    ) -> None:
        """
        this is for the datasets from the MC Dropout repository which were first used
        in probabilistic backpropagation https://arxiv.org/abs/1502.05336
        """
        super(PBPDataset, self).__init__()

        if x is None or y is None or name == "":
            raise ValueError("kwargs needs to have x, y and name for PBP dataset")

        self.x = x
        self.y = y
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


class ClassificationDataset(Dataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(ClassificationDataset, self).__init__()
        self.x: torch.Tensor
        self.y: torch.Tensor

        # these are the parameters of the normalization
        self.mu: torch.Tensor
        self.sigma: torch.Tensor

        # the prototype of the data in this dataset
        self.prototype: torch.Tensor

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]

    def __len__(self) -> int:
        return self.x.size(0)

    def prune(self, idx: torch.Tensor) -> None:
        self.x = self.x[idx]
        self.y = self.y[idx]

    def set_name(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def get_feature_ranges(self) -> torch.Tensor:
        """
        get the feature ranges for all of the x features, this was originally used
        to determine the ranges of features for generating adversarial examples as done by
        deep ensembles paper https://arxiv.org/abs/1612.01474
        """
        return torch.abs(self.x.min(dim=0)[0] - self.x.max(dim=0)[0])

    def valid(self) -> None:
        if torch.any(torch.isinf(self.x)) or torch.any(torch.isnan(self.x)):
            raise ValueError("x has invalid values")
        elif torch.any(torch.isinf(self.y)) or torch.any(torch.isnan(self.y)):
            raise ValueError("y has invalid values")

    def standard_normalize(
        self, mu: Optional[torch.Tensor] = None, sigma: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """standard normalize the dataset by ([x,y] - mu) / sigma"""
        if mu is None or sigma is None:
            self.mu = self.x.mean(dim=0)
            self.sigma = self.x.std(dim=0)
            self.sigma[self.sigma == 0] = 1

            if torch.any(self.sigma == 0):
                raise ValueError(
                    "sigma should not have zero values, see what is going on here"
                )
                self.sigma[self.sigma == 0] = 1

            self.x = (self.x - self.mu) / self.sigma
            if hasattr(self, "prototype"):
                self.prototype = (self.prototype - self.mu) / self.sigma

            self.valid()
            return self.mu, self.sigma

        self.mu = mu
        self.sigma = sigma

        self.x = (self.x - self.mu) / self.sigma
        if hasattr(self, "prototype"):
            self.prototype = (self.prototype - self.mu) / self.sigma

        self.valid()

        return mu, sigma

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        perm = torch.randperm(self.x.size(0))
        return self.x[perm[:n]], self.y[perm[:n]]


def download_data() -> None:
    """check for existence of datasets and download them if they arent in the data dir"""
    if not os.path.exists("data"):
        os.makedirs("data")

    http = PoolManager()
    repo = http.request(
        "GET", "https://github.com/yaringal/DropoutUncertaintyExps/archive/master.zip"
    )

    with zipfile.ZipFile(io.BytesIO(repo.data)) as zip_ref:
        zip_ref.extractall("./data")


def get_pbp_sets(
    name: str, batch_size: int, get_val: bool = True
) -> Tuple[Loader, Optional[Loader], Loader]:
    """
    retrieves the datasets which were used in the following papers
    http://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf
    https://arxiv.org/abs/1502.05336
    https://arxiv.org/abs/1506.02142
    """

    if not os.path.exists("data"):
        print("downloading datasets...")
        download_data()

    if name not in pbp_sets:
        raise ValueError(f"{name} is an unknown pbp dataset")

    # load data
    path = os.path.join(
        "data",
        "DropoutUncertaintyExps-master",
        "UCI_Datasets",
        name,
        "data",
        "data.txt",
    )

    data = torch.from_numpy(np.loadtxt(path)).float()

    # make a random split of train and test
    idx_perm = torch.randperm(data.size(0))
    train_idx = int(data.size(0) * 0.9)

    if not get_val:
        # extract the features and labels
        train_ft = data[idx_perm[:train_idx], :-1]
        train_label = data[idx_perm[:train_idx], -1]

        test_ft = data[idx_perm[train_idx:], :-1]
        test_label = data[idx_perm[train_idx:], -1]

        train = PBPDataset(x=train_ft, y=train_label, name=name)
        test = PBPDataset(x=test_ft, y=test_label, name=name)
        params = train.standard_normalize()
        test.standard_normalize(*params)

        return (
            Loader(train, shuffle=True, batch_size=batch_size),
            None,
            Loader(test, batch_size=batch_size),
        )

    val_n = train_idx // 10

    # extract the features and labels
    train_ft = data[idx_perm[: train_idx - val_n], :-1]
    val_ft = data[idx_perm[train_idx - val_n : train_idx], :-1]

    train_label = data[idx_perm[: train_idx - val_n], -1]
    val_label = data[idx_perm[train_idx - val_n : train_idx], -1]

    test_ft = data[idx_perm[train_idx:], :-1]
    test_label = data[idx_perm[train_idx:], -1]

    train = PBPDataset(x=train_ft, y=train_label, name=name)
    val = PBPDataset(x=val_ft, y=val_label, name=name)
    test = PBPDataset(x=test_ft, y=test_label, name=name)

    params = train.standard_normalize()
    val.standard_normalize(*params)
    test.standard_normalize(*params)

    return (
        Loader(train, shuffle=True, batch_size=batch_size),
        Loader(val, shuffle=True, batch_size=batch_size),
        Loader(test, batch_size=batch_size),
    )
