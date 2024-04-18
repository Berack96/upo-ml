import math as math
import numpy as np

from ml import MLAlgorithm
from learning.data import Dataset

class LinearRegression(MLAlgorithm):
    def __init__(self, dataset:Dataset, learning_rate:float=0.1) -> None:
        self._set_dataset(dataset)

        parameters = dataset.data.shape[1] - 1 #removing the result
        self.theta = np.random.rand(parameters)
        self.alpha = max(0, learning_rate)

    def learning_step(self) -> float:
        theta = self.theta
        alpha = self.alpha
        x, y, m = self._split_data_target(self.learnset)

        self.theta -= alpha * (1/m) * np.sum((x.dot(theta) - y) * x.T, axis=1)
        return self._error(x, y, m)

    def test_error(self) -> float:
        x, y, m = self._split_data_target(self.testset)
        return self._error(x, y, m)

    def _error(self, x:np.ndarray, y:np.ndarray, m:int) -> float:
        diff = (x.dot(self.theta) - y)
        return 1/(2*m) * np.sum(diff ** 2)
