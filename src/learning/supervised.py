import math as math
import numpy as np

from abc import abstractmethod
from learning.ml import MLAlgorithm
from learning.data import Dataset, Data
from learning.functions import cross_entropy_loss, log_loss, lrelu, lrelu_derivative, softmax, softmax_derivative, square_loss, with_bias

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
        h0 = self._h0(x)

        regularization = (self.lambd / m) * self.theta
        regularization[0] = 0

        derivative =  np.mean((h0 - y) * with_bias(x).T, axis=1)
        self.theta -= self.alpha * derivative + regularization
        return self._loss(x, y)

    def _predict_loss(self, dataset:Data) -> float:
        return self._loss(dataset.x, dataset.y)

    def _get_parameters(self):
        return self.theta.copy()

    def _set_parameters(self, parameters):
        self.theta = parameters

    @abstractmethod
    def _loss(self, x:np.ndarray, y:np.ndarray) -> float: pass

class LinearRegression(GradientDescent):
    def _h0(self, x: np.ndarray) -> np.ndarray:
        return self.theta.dot(with_bias(x).T)

    def _loss(self, x:np.ndarray, y:np.ndarray) -> float:
        return square_loss(self._h0(x), y)

class LogisticRegression(GradientDescent):
    def _h0(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-self.theta.dot(with_bias(x).T)))

    def _loss(self, x:np.ndarray, y:np.ndarray) -> float:
        return log_loss(self._h0(x), y)

class MultiLayerPerceptron(MLAlgorithm):
    layers: list[np.ndarray]
    activations: list[np.ndarray]
    previous_delta: list[np.ndarray]
    momentum: float
    learning_rate: float

    def __init__(self, dataset:Dataset, layers:list[int], learning_rate:float=0.1, momentum:float=0.9) -> None:
        super().__init__(dataset)
        input = self._learnset.x.shape[1]
        output = self._learnset.y.shape[1]

        if not all(isinstance(x, int) for x in layers):
            raise Exception("The list of layers must oly be of integers!")
        else: layers.append(output)

        self.layers = []
        self.activations = []
        self.previous_delta = []
        self.momentum = momentum
        self.learning_rate = learning_rate

        for next in layers:
            current = np.random.rand(input + 1, next) * np.sqrt(2 / input) # +1 bias, sqrt is He init
            self.layers.append(current)
            self.previous_delta.append(np.zeros(current.shape))
            input = next

    def _h0(self, x:np.ndarray) -> np.ndarray:
        self.activations = [x]

        for i, layer in enumerate(self.layers):
            x = with_bias(x).dot(layer)
            if i + 1 < len(self.layers): x = lrelu(x)
            self.activations.append(x) # saving activation result
        return softmax(x)

    def _predict_loss(self, dataset:Data) -> float:
        return cross_entropy_loss(self._h0(dataset.x), dataset.y)

    def _learning_step(self) -> float:
        x, y, m, _ = self._learnset.as_tuple()
        delta = softmax_derivative(self._h0(x), y)

        for l in reversed(range(len(self.layers))):
            activation = self.activations[l]
            deltaW = np.dot(with_bias(activation).T, delta) / m
            deltaW *= self.learning_rate
            deltaW += self.momentum * self.previous_delta[l]

            delta = np.dot(delta, self.layers[l][1:].T) # ignoring bias
            delta *= lrelu_derivative(activation)

            self.layers[l] -= deltaW
            self.previous_delta[l] = deltaW

        return self._predict_loss(self._learnset)

    def _get_parameters(self):
        parameters = { 'layers': [], 'previous_delta': [] }
        for x in range(len(self.layers)):
            parameters['layers'].append(self.layers[x].copy())
            parameters['previous_delta'].append(self.previous_delta[x].copy())
        return parameters
    def _set_parameters(self, parameters):
        self.layers = parameters['layers']
        self.previous_delta = parameters['previous_delta']

