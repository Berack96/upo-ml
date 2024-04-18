from abc import ABC, abstractmethod
from learning.data import Dataset

import numpy as np


class MLAlgorithm(ABC):

    dataset: Dataset
    testset: np.ndarray
    learnset: np.ndarray

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
        return (train, test)

    @abstractmethod
    def learning_step(self) -> float:
        pass

    @abstractmethod
    def test_error(self) -> float:
        pass
