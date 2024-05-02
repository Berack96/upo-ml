from abc import ABC, abstractmethod
from plot import Plot
from tqdm import tqdm
from learning.data import Dataset, Data

import numpy as np


class MLAlgorithm(ABC):
    """ Classe generica per gli algoritmi di Machine Learning """

    learnset: Data
    validset: Data
    testset: Data
    _learn_loss: list[float]
    _valid_loss: list[float]

    def __init__(self, dataset:Dataset) -> None:
        learn, test, valid = dataset.get_dataset(0.2, 0.2)
        self.learnset = learn
        self.validset = valid
        self.testset = test

    def learn(self, epochs:int, early_stop:float=0.0000001, max_patience:int=10, verbose:bool=False) -> tuple[int, list, list]:
        learn = []
        valid = []
        count = 0
        patience = 0
        trange = range(epochs)
        if verbose: trange = tqdm(trange, bar_format="Epochs {percentage:3.0f}% [{bar}] {elapsed}{postfix}")

        try:
            for _ in trange:
                if count > 1 and valid[-2] - valid[-1] < early_stop:
                    if patience >= max_patience:
                        self.set_parameters(backup)
                        break
                    patience += 1
                else:
                    backup = self.get_parameters()
                    patience = 0

                count += 1
                learn.append(self.learning_step())
                valid.append(self.validation_loss())

                if verbose: trange.set_postfix({"learn": f"{learn[-1]:2.5f}", "validation": f"{valid[-1]:2.5f}"})
        except KeyboardInterrupt: pass
        if verbose: print(f"Loop ended after {count} epochs")

        self._learn_loss = learn
        self._valid_loss = valid
        return (count, learn, valid)

    def learning_loss(self) -> float:
        return self.predict_loss(self.learnset)

    def validation_loss(self) -> float:
        return self.predict_loss(self.validset)

    def test_loss(self) -> float:
        return self.predict_loss(self.testset)

    def plot(self, skip:int=1000) -> None:
        skip = skip if len(self._learn_loss) > skip else 0
        plot = Plot("Loss", "Time", "Mean Loss")
        plot.line("training", "blue", data=self._learn_loss[skip:])
        plot.line("validation", "red", data=self._valid_loss[skip:])
        plot.wait()

    def confusion_matrix(self, dataset:Data) -> np.ndarray:
        h0 = np.where(self._h0(dataset.x) > 0.5, 1, 0)

        classes = len(np.unique(dataset.y))
        conf_matrix = np.zeros((classes, classes), dtype=int)

        for actual, prediction in zip(dataset.y, h0):
            conf_matrix[int(actual), int(prediction)] += 1
        return conf_matrix

    def accuracy(self, dataset:Data) -> float:
        conf = self.confusion_matrix(dataset)
        correct = np.sum(np.diagonal(conf))
        total = np.sum(conf)
        return correct / total

    @abstractmethod
    def _h0(self, x:np.ndarray) -> np.ndarray: pass
    @abstractmethod
    def learning_step(self) -> float: pass
    @abstractmethod
    def predict_loss(self, dataset:Data) -> float: pass
    @abstractmethod
    def get_parameters(self): pass
    @abstractmethod
    def set_parameters(self, parameters): pass
