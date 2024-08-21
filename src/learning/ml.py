import sys
import numpy as np

from abc import ABC, abstractmethod
from plot import Plot
from tqdm import tqdm
from learning.data import ConfusionMatrix, Dataset, Data, TargetType
from learning.functions import pearson, r_squared

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
        best = (sys.float_info.max, [])
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

                learn_loss = self._learning_step()
                valid_loss = self.validation_loss()
                if valid_loss < best[0]:
                    best = (valid_loss, self._get_parameters())

                if np.isnan(learn_loss) or np.isnan(valid_loss): break
                learn.append(learn_loss)
                valid.append(valid_loss)
                if verbose: trange.set_postfix({"learn": f"{learn[-1]:2.5f}", "validation": f"{valid[-1]:2.5f}"})
        except KeyboardInterrupt: pass
        if verbose: print(f"Loop ended after {count} epochs")

        self._set_parameters(best[1])
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
        plot = Plot("Loss", "Epochs", "Mean Loss")
        plot.line("training", "blue", data=self._learn_loss[skip:])
        plot.line("validation", "red", data=self._valid_loss[skip:])
        plot.wait()

    def display_results(self) -> None:
        print("======== RESULT ========")
        print(f"Loss learn : {self.learning_loss():0.5f}")
        print(f"Loss valid : {self.validation_loss():0.5f}")
        print(f"Loss test  : {self.test_loss():0.5f}")
        print("========================")
        if self._target_type == TargetType.Regression:
            print(f"Pearson    : {self.test_pearson():0.5f}")
            print(f"R^2        : {self.test_r_squared():0.5f}")
            print("========================")
        elif self._target_type != TargetType.NoTarget:
            conf = self.test_confusion_matrix()
            conf.print()
            print("========================")

    def test_confusion_matrix(self) -> ConfusionMatrix:
        if self._target_type != TargetType.Classification\
        and self._target_type != TargetType.MultiClassification:
            return None

        h0 = self._h0(self._testset.x)
        y = self._testset.y
        if h0.ndim == 1:
            h0 = np.where(h0 > 0.5, 1, 0)

        return ConfusionMatrix(y, h0)

    def test_pearson(self) -> float:
        if self._target_type != TargetType.Regression:
            return 0
        return pearson(self._h0(self._testset.x), self._testset.y)

    def test_r_squared(self) -> float:
        if self._target_type != TargetType.Regression:
            return 0
        return r_squared(self._h0(self._testset.x), self._testset.y)

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
