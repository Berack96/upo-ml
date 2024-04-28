import pandas as pd
import numpy as np

from typing_extensions import Self

class Dataset:
    def __init__(self, csv:str, target:str, classification:bool=None) -> None:
        data = pd.read_csv(csv)

        # move target to the start
        col_target = data.pop(target)
        data.insert(0, target, col_target)
        data.insert(1, "Bias", 1.0)

        if classification == None:
            classification = (data[target].dtype == object)

        self.original = data
        self.data = data
        self.target = target
        self.classification = classification

    def remove(self, columns:list[str]) -> Self:
        for col in columns:
            self.data.pop(col)
        return self

    def regularize(self, excepts:list[str]=[]) -> Self:
        excepts.append(self.target)
        excepts.append("Bias")
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

    def to_numbers(self, columns:list[str]=[]) -> Self:
        data = self.data
        for col in columns:
            if data[col].dtype == object:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        return self

    def handle_na(self) -> Self:
        self.data = self.data.dropna()
        return self

    def shuffle(self) -> Self:
        self.data = self.data.sample(frac=1)
        return self

    def as_ndarray(self) -> np.ndarray:
        return self.data.to_numpy()

    def get_index(self, column:str) -> int:
        return self.data.columns.get_loc(column)

class PrincipalComponentAnalisys:
    def __init__(self, data:np.ndarray) -> None:
        self.data = data

    def reduce(self, total:int=0, threshold:float=1) -> Self:
        columns = self.data.shape[1]
        if total > columns or total <= 0:
            total = columns
        if threshold <= 0 or threshold > 1:
            threshold = 1



if __name__ == "__main__":
    df = Dataset("datasets\\regression\\automobile.csv", "symboling")
    attributes_to_modify = ["fuel-system", "engine-type", "drive-wheels", "body-style", "make", "engine-location", "aspiration", "fuel-type", "num-of-cylinders", "num-of-doors"]
    df.factorize(attributes_to_modify)
    df.to_numbers(["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm", "price"])
    df.handle_na()
    df.regularize(excepts=attributes_to_modify)
    print(df.data.dtypes)
