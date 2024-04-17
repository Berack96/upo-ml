import pandas as pd
import numpy as np

from typing_extensions import Self

class Dataset:
    def __init__(self, csv:str, target:str) -> None:
        data = pd.read_csv(csv)

        # move target to the start
        col_target = data.pop(target)
        data.insert(0, target, col_target)

        self.data = data
        self.target = target
        self.classification = (data[target].dtype == object)

    def regularize(self, excepts:list=[]) -> Self:
        excepts.append(self.target)
        for col in self.data:
            if col not in excepts:
                dt = self.data[col]
                self.data[col] = (dt - dt.mean()) / dt.std()
        return self

    def factorize(self, columns:list=[]) -> Self:
        data = self.data
        for col in columns:
            data[col] = pd.factorize(data[col])[0]
        return self

    def to_numbers(self, columns:list=[]) -> Self:
        data = self.data
        for col in self.data.columns:
            if data[col].dtype == object:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        return self

    def handle_na(self) -> Self:
        self.data = self.data.dropna()
        return self

    def shuffle(self) -> Self:
        self.data = self.data.sample(frac=1)
        return self

    def as_ndarray(self, bias=True):
        data = self.data.copy()
        if bias: data.insert(1, "Bias", 1.0)
        return data.to_numpy()

class PrincipalComponentAnalisys:
    def __init__(self, data:np.ndarray) -> None:
        self.data = data

    def reduce(self, total:int=0, threshold:float=1) -> Self:
        columns = self.data.shape[1]
        if total > columns or total <= 0:
            total = columns
        if threshold <= 0 or threshold > 1:
            threshold = 1

