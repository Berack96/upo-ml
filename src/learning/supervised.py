import math as math
import numpy as np

from learning.ml import MLRegression
from learning.data import Dataset

class LinearRegression(MLRegression):
    theta:np.ndarray
    alpha:float

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

    def predict_loss(self, dataset:np.ndarray) -> float:
        x, y, m = self._split_data_target(dataset)
        return self._error(x, y, m)

    def _error(self, x:np.ndarray, y:np.ndarray, m:int) -> float:
        diff = (x.dot(self.theta) - y)
        return 1/(2*m) * np.sum(diff ** 2)

    def get_parameters(self):
        return self.theta.copy()

    def set_parameters(self, parameters):
        self.theta = parameters
