import numpy as np
import pandas as pd

from enum import Enum
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

    def normalize(self, excepts:list[str]=[]) -> Self:
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

        seed = np.random.randint(0, 4294967295)
        rng = np.random.default_rng(seed)
        data = []
        for x in splitted:
            samples = rng.choice(x, size=total_each, replace=True, shuffle=False)
            data.append(samples)

        return np.concatenate(data, axis=0)

    def split_data_target(self, data:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target = data[:, 0] if self.target_type != TargetType.NoTarget else None
        data = data[:, 1:]
        if self.target_type == TargetType.MultiClassification:
            target = target.astype(int)
            uniques = np.unique(target).shape[0]
            target = np.eye(uniques)[target]
        return (data, target)

    def get_dataset(self, test_frac:float=0.2, valid_frac:float=0.2) -> tuple[Data, Data, Data]:
        data = self.data.to_numpy()
        data = self.prepare_classification(data)

        np.random.shuffle(data)

        total = data.shape[0]
        valid_cutoff = int(total * valid_frac)
        test_cutoff = int(total * test_frac) + valid_cutoff

        valid = data[:valid_cutoff]
        test = data[valid_cutoff:test_cutoff]
        learn = data[test_cutoff:]

        l = []
        for data in [learn, test, valid]:
            data, target = self.split_data_target(data)
            l.append(Data(data, target))
        return l

class ConfusionMatrix:
    matrix:np.ndarray

    def __init__(self, dataset_y: np.ndarray, predictions_y:np.ndarray) -> None:
        classes = len(np.unique(dataset_y))
        conf_matrix = np.zeros((classes, classes), dtype=int)

        for actual, prediction in zip(dataset_y, predictions_y):
            conf_matrix[int(actual), int(prediction)] += 1

        self.matrix = conf_matrix
        self.classes = classes
        self.total = dataset_y.shape[0]
        self.tp = np.diagonal(conf_matrix)
        self.fp = np.sum(conf_matrix, axis=0) - self.tp
        self.fn = np.sum(conf_matrix, axis=1) - self.tp
        self.tn = self.total - (self.tp + self.fp + self.fn)

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

    def f1_score_per_class(self) -> np.ndarray:
        prec = self.precision_per_class()
        rec = self.recall_per_class()
        return self.divide_ignore_zero(2 * prec * rec, prec + rec)

    def specificity_per_class(self) -> np.ndarray:
        return self.divide_ignore_zero(self.tn, self.tn + self.fp)

    def accuracy(self) -> float:
        return self.tp.sum() / self.total

    def precision(self) -> float:
        precision_per_class = self.precision_per_class()
        support = np.sum(self.matrix, axis=1)
        return np.average(precision_per_class, weights=support)

    def recall(self) -> float:
        recall_per_class = self.recall_per_class()
        support = np.sum(self.matrix, axis=1)
        return np.average(recall_per_class, weights=support)

    def f1_score(self) -> float:
        f1_per_class = self.f1_score_per_class()
        support = np.sum(self.matrix, axis=1)
        return np.average(f1_per_class, weights=support)

    def specificity(self) -> float:
        specificity_per_class = self.specificity_per_class()
        support = np.sum(self.matrix, axis=1)
        return np.average(specificity_per_class, weights=support)
