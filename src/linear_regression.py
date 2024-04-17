import math as math
import numpy as np
import matplotlib.pyplot as plt

from data import Dataset


class LinearRegression:
    def __init__(self, dataset:Dataset, learning_rate:float=0.1) -> None:
        ndarray = dataset.shuffle().as_ndarray()
        parameters = ndarray.shape[1] - 1 #removing the result

        split = int(ndarray.shape[0] * 0.2)
        self.testset = ndarray[split:]
        self.trainingset = ndarray[:split]

        self.theta = np.random.rand(parameters)
        self.alpha = max(0, learning_rate)

    def learn(self, times:int) -> list:
        train = []
        test = []
        for _ in range(0, max(1, times)):
            train.append(self.learning_step())
            test.append(self.test_error())
        return (train, test)

    def learning_step(self) -> float:
        theta = self.theta
        alpha = self.alpha
        x = np.delete(self.trainingset, 0, 1)
        y = self.trainingset[:, 0]
        m = self.trainingset.shape[0]

        diff = (x.dot(theta) - y)
        sum = np.sum(diff * x.T, axis=1)
        theta -= alpha * (1/m) * sum
        self.theta = theta
        return self._error(x, y, m)

    def test_error(self) -> float:
        x = np.delete(self.testset, 0, 1)
        y = self.testset[:, 0]
        m = self.testset.shape[0]
        return self._error(x, y, m)

    def _error(self, x:np.ndarray, y:np.ndarray, m:int) -> float:
        diff = (x.dot(self.theta) - y)
        return 1/(2*m) * np.sum(diff ** 2)

def auto_mpg(epoch:int):
    df = Dataset("datasets\\auto-mpg.csv", "MPG")

    df.to_numbers(["HP"])
    df.handle_na()
    df.regularize(excepts=["Cylinders","Year","Origin"])

    lr = LinearRegression(df, learning_rate=0.0001)
    return lr.learn(epoch)

def automobile(epoch:int):
    df = Dataset("datasets\\regression\\automobile.csv", "symboling")

    attributes_to_modify = ["fuel-system", "engine-type", "drive-wheels", "body-style", "make", "engine-location", "aspiration", "fuel-type", "num-of-cylinders", "num-of-doors"]
    df.factorize(attributes_to_modify)
    df.to_numbers()
    df.handle_na()
    df.regularize(excepts=attributes_to_modify)

    lr = LinearRegression(df, learning_rate=0.001)
    return lr.learn(epoch)


if __name__ == '__main__':
    epoch = 10000
    skip = - int(epoch * 0.9)
    err_train, err_test = auto_mpg(epoch)
    plt.title("Error")
    plt.xlabel("Time")
    plt.ylabel("Mean Error")
    plt.plot(err_train[skip:-1], color="red")
    plt.plot(err_test[skip:-1], color="blue")
    plt.show()
