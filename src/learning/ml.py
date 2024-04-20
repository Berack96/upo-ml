from abc import ABC, abstractmethod
from learning.data import Dataset
from plot import Plot

import numpy as np


class MLAlgorithm(ABC):
    """ Classe generica per gli algoritmi di Machine Learning """

    dataset: Dataset
    testset: np.ndarray
    learnset: np.ndarray
    test_error: list[float]
    train_error: list[float]

    def _set_dataset(self, dataset:Dataset, split:float=0.2):
        ndarray = dataset.shuffle().as_ndarray()
        split = int(ndarray.shape[0] * split)

        self.dataset = dataset
        self.testset = ndarray[split:]
        self.learnset = ndarray[:split]

    def _split_data_target(self, dset:np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        x = np.delete(dset, 0, 1)
        y = dset[:, 0]
        m = dset.shape[0]
        return (x, y, m)

    def learn(self, times:int) -> tuple[list, list]:
        train = []
        test = []
        for _ in range(0, max(1, times)):
            train.append(self.learning_step())
            test.append(self.test_error())

        self.train_error = train
        self.test_error = test
        return (train, test)

    @abstractmethod
    def learning_step(self) -> float:
        pass

    @abstractmethod
    def test_error(self) -> float:
        pass

    @abstractmethod
    def plot(self, skip:int=1000) -> None:
        pass


class MLRegression(MLAlgorithm):
    def plot(self, skip:int=1000) -> None:
        plot = Plot("Error", "Time", "Mean Error")
        plot.line("training", "red", data=self.train_error[skip:])
        plot.line("test", "blue", data=self.test_error[skip:])
        plot.wait()
