import numpy as np
import pandas as pd

from enum import Enum
import sklearn
import sklearn.metrics
from typing_extensions import Self

class TargetType(Enum):
    Regression = 1
    Classification = 2
    MultiClassification = 3
    NoTarget = 4

class Data:
    x: np.ndarray
    y: np.ndarray
    size: int
    param: int

    def __init__(self, x:np.ndarray, y:np.ndarray) -> None:
        self.x = x
        self.y = y
        self.size = x.shape[0]
        self.param = x.shape[1]
    def __str__(self) -> str:
        return "X: " + str(self.x) + "\nY: " + str(self.y)
    def as_tuple(self) -> tuple[np.ndarray, np.ndarray, int, int]:
        return (self.x, self.y, self.size, self.param)

class Dataset:
    data: pd.DataFrame
    target: str
    target_type: TargetType

    def __init__(self, csv:str, target:str, target_type:TargetType) -> None:
        self.original = pd.read_csv(csv)
        self.data = self.original
        self.target = target
        self.target_type = target_type

        # move target to the start
        col_target = self.data.pop(target)
        self.data.insert(0, target, col_target)

    def remove(self, columns:list[str]) -> Self:
        for col in columns:
            self.data.pop(col)
        return self

    def standardize(self, excepts:list[str]=[]) -> Self:
        if excepts is None: excepts = []
        else: excepts.append(self.target)

        for col in self.data:
            if col not in excepts:
                datacol = self.data[col]
                self.data[col] = (datacol - datacol.mean()) / datacol.std()
        return self

    def factorize(self, columns:list[str]=[]) -> Self:
        data = self.data
        for col in columns:
            data[col] = pd.factorize(data[col])[0]
        return self

    def numbers(self, columns:list[str]=[]) -> Self:
        data = self.data
        for col in columns:
            if data[col].dtype == object:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        return self

    def handle_na(self) -> Self:
        self.data = self.data.dropna()
        return self

    def prepare_classification(self, data:np.ndarray) -> np.ndarray:
        if self.target_type == TargetType.Regression or self.target_type == TargetType.NoTarget:
            return data

        classes = np.unique(data[:, 0])
        splitted = [data[ data[:,0] == k ] for k in classes ]
        total_each = np.average([len(x) for x in splitted]).astype(int)

        data = []
        for x in splitted:
            total = total_each - x.shape[0]
            data.append(x)
            if total > 0:
                samples = np.random.choice(x, size=total, replace=True)
                data.append(samples)

        return np.concatenate(data, axis=0)

    def split_data_target(self, data:np.ndarray) -> Data:
        target = data[:, 0] if self.target_type != TargetType.NoTarget else None
        data = data[:, 1:]
        if self.target_type == TargetType.MultiClassification:
            target = target.astype(int)
            uniques = np.unique(target).shape[0]
            target = np.eye(uniques)[target]
        return Data(data, target)

    def split_dataset(self, data:np.ndarray, valid_frac:float, test_frac:float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        total = data.shape[0]
        valid_cutoff = int(total * valid_frac)
        test_cutoff = int(total * test_frac) + valid_cutoff

        learn = data[test_cutoff:]
        valid = data[:valid_cutoff]
        test = data[valid_cutoff:test_cutoff]
        return (learn, valid, test)


    def get_dataset(self, test_frac:float=0.2, valid_frac:float=0.2) -> tuple[Data, Data, Data]:
        data = self.data.to_numpy()
        max_iter = 10
        while max_iter > 0:
            max_iter -= 1
            try:
                np.random.shuffle(data)
                learn, valid, test = self.split_dataset(data, valid_frac, test_frac)

                if self.target_type == TargetType.Regression or self.target_type == TargetType.NoTarget:
                    learn = self.prepare_classification(learn)
                    valid = self.prepare_classification(valid)
                    test = self.prepare_classification(test)

                learn = self.split_data_target(learn)
                valid = self.split_data_target(valid)
                test = self.split_data_target(test)
                return (learn, valid, test)
            except:
                if max_iter == 0:
                    raise Exception("Could not split dataset evenly for the classes, try again with another seed or add more cases in the dataset")

class ConfusionMatrix:
    matrix:np.ndarray

    def __init__(self, dataset_y: np.ndarray, predictions_y:np.ndarray) -> None:
        if len(dataset_y.shape) > 1:
            dataset_y = np.argmax(dataset_y, axis=1)
        if len(predictions_y.shape) > 1:
            predictions_y = np.argmax(predictions_y, axis=1)

        classes = len(np.unique(dataset_y))
        conf_matrix = np.zeros((classes, classes), dtype=int)

        for actual, prediction in zip(dataset_y, predictions_y):
            conf_matrix[int(actual), int(prediction)] += 1

        self.matrix = conf_matrix
        self.classes = classes
        self.total = dataset_y.shape[0]
        self.weights = np.sum(conf_matrix, axis=1)
        self.tp = np.diagonal(conf_matrix)
        self.fp = np.sum(conf_matrix, axis=0) - self.tp
        self.fn = np.sum(conf_matrix, axis=1) - self.tp
        self.tn = self.total - (self.tp + self.fp + self.fn)
        self.kappa = sklearn.metrics.cohen_kappa_score(dataset_y, predictions_y)

    def divide_ignore_zero(self, a:np.ndarray, b:np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(a, b)
            c[c == np.inf] = 0
            return np.nan_to_num(c)

    def accuracy_per_class(self) -> np.ndarray:
        return self.tp / np.sum(self.matrix, axis=1)

    def precision_per_class(self) -> np.ndarray:
        return self.divide_ignore_zero(self.tp, self.tp + self.fp)

    def recall_per_class(self) -> np.ndarray:
        return self.divide_ignore_zero(self.tp, self.tp + self.fn)

    def specificity_per_class(self) -> np.ndarray:
        return self.divide_ignore_zero(self.tn, self.tn + self.fp)

    def f1_score_per_class(self) -> np.ndarray:
        prec = self.precision_per_class()
        rec = self.recall_per_class()
        return self.divide_ignore_zero(2 * prec * rec, prec + rec)

    def accuracy(self) -> float:
        return self.tp.sum() / self.total

    def precision(self) -> float:
        precision_per_class = self.precision_per_class()
        return np.average(precision_per_class, weights=self.weights)

    def recall(self) -> float:
        recall_per_class = self.recall_per_class()
        return np.average(recall_per_class, weights=self.weights)

    def specificity(self) -> float:
        specificity_per_class = self.specificity_per_class()
        return np.average(specificity_per_class, weights=self.weights)

    def f1_score(self) -> float:
        f1_per_class = self.f1_score_per_class()
        return np.average(f1_per_class, weights=self.weights)

    def cohen_kappa(self) -> float:
        return self.kappa

    def print(self)-> None:
        print(f"Cohen Kappa: {self.cohen_kappa():0.5f}")
        print(f"Accuracy   : {self.accuracy():0.5f} - classes {self.accuracy_per_class()}")
        print(f"Precision  : {self.precision():0.5f} - classes {self.precision_per_class()}")
        print(f"Recall     : {self.recall():0.5f} - classes {self.recall_per_class()}")
        print(f"Specificity: {self.specificity():0.5f} - classes {self.specificity_per_class()}")
        print(f"F1 score   : {self.f1_score():0.5f} - classes {self.f1_score_per_class()}")
