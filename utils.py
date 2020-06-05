from typing import Tuple

import numpy as np  # type: ignore


class ClassStats:
    def __init__(self) -> None:
        self.accuracy = np.array([])

    def set(self, acc: float) -> None:
        self.accuracy = np.append(self.accuracy, acc)

    def get_acc(self) -> Tuple[float, float]:
        return self.accuracy.mean(), self.accuracy.std() * 2

    def __str__(self) -> str:
        return f"acc: {self.get_acc()[0]}"


class Stats:
    def __init__(self) -> None:
        self.ll = np.array([])
        self.rmse = np.array([])
        self.aleatoric = np.array([])
        self.epistemic = np.array([])
        self.n = 1

    def set(self, ll: float, rmse: float, aleatoric: float, epistemic: float) -> None:
        self.ll = np.append(self.ll, ll)
        self.rmse = np.append(self.rmse, rmse)
        self.aleatoric = np.append(self.aleatoric, aleatoric)
        self.epistemic = np.append(self.epistemic, epistemic)

    def get_uncertainties(self) -> Tuple[float, float]:
        return self.aleatoric.mean(), self.epistemic.mean()

    def get_rmse(self) -> Tuple[float, float]:
        return self.rmse.mean(), self.rmse.std() * 2

    def get_ll(self) -> Tuple[float, float]:
        return self.ll.mean(), self.ll.std() * 2

    def __str__(self) -> str:
        return f"ll: {self.get_ll()[0]} rmse: {self.get_rmse()[0]}"
