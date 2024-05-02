import pandas as pd
import numpy as np

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
        excepts.append(self.target)
        for col in self.data:
            if col not in excepts:
                index = self.data.columns.get_loc(col)
                datacol = self.data.pop(col)
                datacol = (datacol - datacol.mean()) / datacol.std()
                self.data.insert(index, col, datacol)
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

    def get_dataset(self, test_frac:float=0.15, valid_frac:float=0.15) -> tuple[Data, Data, Data]:
        data = self.data.to_numpy()
        data = np.insert(data, 1, 1, axis=1) # adding bias
        np.random.shuffle(data)

        total = data.shape[0]
        valid_cutoff = int(total * valid_frac)
        test_cutoff = int(total * test_frac) + valid_cutoff

        valid = data[:valid_cutoff]
        test = data[valid_cutoff:test_cutoff]
        learn = data[test_cutoff:]

        l = []
        for ds in [learn, test, valid]:
            target = ds[:, 0] if self.target_type != TargetType.NoTarget else None
            ds = ds[:, 1:]
            if self.target_type == TargetType.MultiClassification:
                target = target.astype(int)
                uniques = np.unique(target).shape[0]
                target = np.eye(uniques)[target]
            l.append(Data(ds, target))
        return l

if __name__ == "__main__":
    ds = Dataset("datasets\\classification\\frogs.csv", "Species", TargetType.MultiClassification)
    ds.remove(["Family", "Genus", "RecordID"])
    ds.factorize(["Species"])

    np.random.seed(0)
    learn, test, valid = ds.get_dataset()
    print(learn)
    print(test)
    print(valid)

