from abc import ABC, abstractmethod
from learning.data import Dataset
from plot import Plot
from tqdm import tqdm

import numpy as np


class MLAlgorithm(ABC):
    """ Classe generica per gli algoritmi di Machine Learning """

    dataset: Dataset
    testset: np.ndarray
    learnset: np.ndarray
    _valid_loss: list[float]
    _train_loss: list[float]

    def _set_dataset(self, dataset:Dataset, split:float=0.2):
        ndarray = dataset.shuffle().as_ndarray()
        splitT = int(ndarray.shape[0] * split)
        splitV = int(splitT / 2)

        self.dataset = dataset
        self.validset = ndarray[:splitV]
        self.testset = ndarray[splitV:splitT]
        self.learnset = ndarray[splitT:]

    def _split_data_target(self, dset:np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        x = np.delete(dset, 0, 1)
        y = dset[:, 0]
        m = dset.shape[0]
        return (x, y, m)

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

        self._train_loss = learn
        self._valid_loss = valid
        return (count, learn, valid)

    def learning_loss(self) -> float:
        return self.predict_loss(self.learnset)

    def validation_loss(self) -> float:
        return self.predict_loss(self.validset)

    def test_loss(self) -> float:
        return self.predict_loss(self.testset)


    @abstractmethod
    def learning_step(self) -> float: pass
    @abstractmethod
    def predict_loss(self, dataset:np.ndarray) -> float: pass
    @abstractmethod
    def get_parameters(self): pass
    @abstractmethod
    def set_parameters(self, parameters): pass

    @abstractmethod
    def plot(self, skip:int=1000) -> None:
        skip = skip if len(self._train_loss) > skip else 0
        plot = Plot("Loss", "Time", "Mean Loss")
        plot.line("training", "blue", data=self._train_loss[skip:])
        plot.line("validation", "red", data=self._valid_loss[skip:])
        plot.wait()



class MLRegression(MLAlgorithm):
    def plot(self, skip: int = 1000) -> None:
        return super().plot(skip)

class MLClassification(MLAlgorithm):
    def plot(self, skip: int = 1000) -> None:
        return super().plot(skip)