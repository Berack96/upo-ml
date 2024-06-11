from abc import ABC, abstractmethod
from plot import Plot
from tqdm import tqdm
from learning.data import ConfusionMatrix, Dataset, Data, TargetType

import numpy as np


class MLAlgorithm(ABC):
    """ Classe generica per gli algoritmi di Machine Learning """
    _target_type: TargetType
    _learnset: Data
    _validset: Data
    _testset: Data
    _learn_loss: list[float]
    _valid_loss: list[float]

    def __init__(self, dataset:Dataset) -> None:
        learn, test, valid = dataset.get_dataset()
        self._target_type = dataset.target_type
        self._learnset = learn
        self._validset = valid
        self._testset = test

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
                        self._set_parameters(backup)
                        break
                    patience += 1
                else:
                    backup = self._get_parameters()
                    patience = 0

                count += 1
                learn.append(self._learning_step())
                valid.append(self.validation_loss())

                if verbose: trange.set_postfix({"learn": f"{learn[-1]:2.5f}", "validation": f"{valid[-1]:2.5f}"})
        except KeyboardInterrupt: pass
        if verbose: print(f"Loop ended after {count} epochs")

        self._learn_loss = learn
        self._valid_loss = valid
        return (count, learn, valid)

    def learning_loss(self) -> float:
        return self._predict_loss(self._learnset)

    def validation_loss(self) -> float:
        return self._predict_loss(self._validset)

    def test_loss(self) -> float:
        return self._predict_loss(self._testset)

    def plot(self, skip:int=1000) -> None:
        skip = skip if len(self._learn_loss) > skip else 0
        plot = Plot("Loss", "Time", "Mean Loss")
        plot.line("training", "blue", data=self._learn_loss[skip:])
        plot.line("validation", "red", data=self._valid_loss[skip:])
        plot.wait()

    def display_results(self) -> None:
        print("======== RESULT ========")
        print(f"Loss learn : {self.learning_loss():0.5f}")
        print(f"Loss valid : {self.validation_loss():0.5f}")
        print(f"Loss test  : {self.test_loss():0.5f}")
        if self._target_type == TargetType.Regression:
            print(f"R^2        : {self.test_r_squared():0.5f}")
        else:
            conf = self.test_confusion_matrix()
            print(f"Accuracy   : {conf.accuracy_per_class()}")
            print(f"Precision  : {conf.precision_per_class()}")
            print(f"Recall     : {conf.recall_per_class()}")
            print(f"F1 score   : {conf.f1_score_per_class()}")
            print(f"Specificity: {conf.specificity_per_class()}")

    def test_confusion_matrix(self) -> ConfusionMatrix:
        if self._target_type != TargetType.Classification\
        and self._target_type != TargetType.MultiClassification:
            return None

        h0 = np.where(self._h0(self._testset.x) > 0.5, 1, 0)
        return ConfusionMatrix(self._testset.y, h0)

    def test_r_squared(self) -> float:
        if self._target_type != TargetType.Regression:
            return 0

        h0 = self._h0(self._testset.x)
        y_mean = np.mean(self._testset.y)
        ss_total = np.sum((self._testset.y - y_mean) ** 2)
        ss_resid = np.sum((self._testset.y - h0) ** 2)
        return 1 - (ss_resid / ss_total)

    @abstractmethod
    def _h0(self, x:np.ndarray) -> np.ndarray: pass
    @abstractmethod
    def _learning_step(self) -> float: pass
    @abstractmethod
    def _predict_loss(self, dataset:Data) -> float: pass
    @abstractmethod
    def _get_parameters(self): pass
    @abstractmethod
    def _set_parameters(self, parameters): pass
