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
        self.theta = np.random.rand(self._learnset.param + 1) # bias
        self.alpha = max(0, learning_rate)
        self.lambd = max(0, regularization)

    def _learning_step(self) -> float:
        x, y, m, _ = self._learnset.as_tuple()

        regularization = (self.lambd / m) * self.theta
        regularization[0] = 0
        derivative = self.alpha * (1/m) * np.sum((self._h0(x) - y) * self.with_bias(x).T, axis=1)
        self.theta -= derivative + regularization
        return self._loss(x, y, m)

    def _predict_loss(self, dataset:Data) -> float:
        return self._loss(dataset.x, dataset.y, dataset.size)

    def _get_parameters(self):
        return self.theta.copy()

    def _set_parameters(self, parameters):
        self.theta = parameters

    @abstractmethod
    def _loss(self, x:np.ndarray, y:np.ndarray, m:int) -> float: pass


class LinearRegression(GradientDescent):
    def _h0(self, x: np.ndarray) -> np.ndarray:
        return self.theta.dot(self.with_bias(x).T)

    def _loss(self, x:np.ndarray, y:np.ndarray, m:int) -> float:
        diff = (self._h0(x) - y)
        return 1/(2*m) * np.sum(diff ** 2)

class LogisticRegression(GradientDescent):
    def _h0(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-self.theta.dot(self.with_bias(x).T)))

    def _loss(self, x:np.ndarray, y:np.ndarray, m:int) -> float:
        not_zero = 1e-15
        h0 = self._h0(x)
        diff = - y*np.log(h0 + not_zero) - (1-y)*np.log(1-h0 + not_zero)
        return 1/m * np.sum(diff)

class MultiLayerPerceptron(MLAlgorithm):
    layers: list[np.ndarray]
    activations: list[np.ndarray]

    def __init__(self, dataset:Dataset, layers:list[int]) -> None:
        super().__init__(dataset)
        input = self._learnset.x.shape[1]
        output = self._learnset.y.shape[1]

        if type(layers) is not list[int]:
            layers = [4, 3, output]
        else: layers.append(output)

        self.layers = []
        self.activations = []

        for next in layers:
            current = np.random.rand(input + 1, next) # +1 bias
            self.layers.append(current)
            input = next

    def _h0(self, x:np.ndarray) -> np.ndarray:
        self.activations = [x]

        for layer in self.layers:
            x = self.with_bias(x)
            x = x.dot(layer)
            x = x * (x > 0) # activation function ReLU
            self.activations.append(x) # saving activation result
        return self.softmax(x)

    def _learning_step(self) -> float:
        x, y, m, _ = self._learnset.as_tuple()
        delta = self._h0(x) - y # first term is derivative of softmax

        for l in reversed(range(len(self.layers))):
            activation = self.activations[l]
            deltaW = np.dot(self.with_bias(activation).T, delta) / m

            if l > 0:
                delta = np.dot(delta, self.layers[l][:-1].T) # ignoring bias
                delta[activation <= 0] = 0 # derivative ReLU
            self.layers[l] -= deltaW

        return self._predict_loss(self._learnset)

    def softmax(self, input:np.ndarray) -> np.ndarray:
        input = input - np.max(input, axis=1, keepdims=True) # for overflow
        exp_input = np.exp(input)
        total_sum = np.sum(exp_input, axis=1, keepdims=True)
        return exp_input / total_sum

    def _predict_loss(self, dataset:Data) -> float:
        diff = self._h0(dataset.x) - dataset.y
        return 1/(2*dataset.size) * np.sum(diff ** 2)


    def _get_parameters(self):
        parameters = []
        for x in self.layers:
            parameters.append(x.copy())
        return parameters
    def _set_parameters(self, parameters):
        self.layers = parameters

