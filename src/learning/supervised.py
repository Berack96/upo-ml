import math as math
import numpy as np

from abc import abstractmethod
from learning.ml import MLAlgorithm
from learning.data import Dataset, Data

class GradientDescent(MLAlgorithm):
    theta:np.ndarray
    alpha:float
    lambd:float

    def __init__(self, dataset:Dataset, learning_rate:float=0.1, regularization:float=0.01) -> None:
        super().__init__(dataset)
        self.theta = np.random.rand(self.learnset.param)
        self.alpha = max(0, learning_rate)
        self.lambd = max(0, regularization)

    def learning_step(self) -> float:
        x, y, m, _ = self.learnset.as_tuple()

        regularization = (self.lambd / m) * self.theta
        regularization[0] = 0
        derivative = self.alpha * (1/m) * np.sum((self._h0(x) - y) * x.T, axis=1)
        self.theta -= derivative + regularization
        return self._loss(x, y, m)

    def predict_loss(self, dataset:Data) -> float:
        return self._loss(dataset.x, dataset.y, dataset.size)

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
        diff = (self._h0(x) - y)
        return 1/(2*m) * np.sum(diff ** 2)

class LogisticRegression(GradientDescent):
    def _h0(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-self.theta.dot(x.T)))

    def _loss(self, x:np.ndarray, y:np.ndarray, m:int) -> float:
        h0 = self._h0(x)
        diff = -y*np.log(h0) -(1-y)*np.log(1-h0)
        return 1/m * np.sum(diff)

class MultiLayerPerceptron(MLAlgorithm):
    layers: list[np.ndarray]
    calculated: list[np.ndarray]

    def __init__(self, dataset:Dataset, layers:list[int]) -> None:
        super().__init__(dataset)
        input = self.learnset.x.shape[1]
        output = self.learnset.y.shape[1]

        if type(layers) is not list[int]:
            layers = [4, 3, output]
        else: layers.append(output)

        self.layers = []
        self.calculated = []

        for next in layers:
            current = np.random.rand(input, next)
            self.layers.append(current)
            input = next + 1 # bias

    def _h0(self, x:np.ndarray) -> np.ndarray:
        input = x
        for i, layer in enumerate(self.layers):
            if i != 0:
                ones = np.ones(shape=(input.shape[0], 1))
                input = np.hstack([input, ones])
            input = input.dot(layer)
            input = input * (input > 0) # activation function ReLU
            self.calculated[i] = input # saving previous result
        return self.soft_max(input)

    def soft_max(self, input:np.ndarray) -> np.ndarray:
        input = np.exp(input)
        total_sum = np.sum(input, axis=1)
        input = input.T / total_sum
        return input.T

    def learning_step(self) -> float:

        raise NotImplemented

    def predict_loss(self, dataset:Data) -> float:
        diff = self._h0(dataset.x) - dataset.y
        return 1/(2*dataset.size) * np.sum(diff ** 2)


    def get_parameters(self):
        parameters = []
        for x in self.layers:
            parameters.append(x.copy())
        return parameters
    def set_parameters(self, parameters):
        self.layers = parameters

