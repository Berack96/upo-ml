import math as math
import numpy as np

from abc import abstractmethod
from learning.ml import MLAlgorithm

class GradientDescent(MLAlgorithm):
    theta:np.ndarray
    alpha:float

    def __init__(self, dataset:np.ndarray, learning_rate:float=0.1) -> None:
        self._set_dataset(dataset)

        parameters = dataset.shape[1] - 1 #removing the result
        self.theta = np.random.rand(parameters)
        self.alpha = max(0, learning_rate)

    def learning_step(self) -> float:
        x, y, m = self._split_data_target(self.learnset)

        self.theta -= self.alpha * (1/m) * np.sum((self._h0(x) - y) * x.T, axis=1)
        return self._loss(x, y, m)

    def predict_loss(self, dataset:np.ndarray) -> float:
        x, y, m = self._split_data_target(dataset)
        return self._loss(x, y, m)

    def get_parameters(self):
        return self.theta.copy()

    def set_parameters(self, parameters):
        self.theta = parameters

    @abstractmethod
    def _loss(self, x:np.ndarray, y:np.ndarray, m:int) -> float: pass


class LinearRegression(GradientDescent):
    def _h0(self, x: np.ndarray) -> np.ndarray:
        return self.theta.dot(x.T)

    def _loss(self, x:np.ndarray, y:np.ndarray, m:int) -> float:
        diff = (x.dot(self.theta) - y)
        return 1/(2*m) * np.sum(diff ** 2)

class LogisticRegression(GradientDescent):
    def _h0(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-self.theta.dot(x.T)))

    def _loss(self, x:np.ndarray, y:np.ndarray, m:int) -> float:
        h0 = self._h0(x)
        diff = -y*np.log(h0) -(1-y)*np.log(1-h0)
        return 1/m * np.sum(diff)
