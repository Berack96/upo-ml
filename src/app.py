from learning.data import Dataset, TargetType
from learning.supervised import LinearRegression, LogisticRegression, MultiLayerPerceptron
from learning.ml import MLAlgorithm
from typing import Callable

DATASET = "datasets/"
REGRESSION = DATASET + "regression/"
CLASSIFICATION = DATASET + "classification/"

def auto_mpg() -> tuple[int, MLAlgorithm]:
    ds = Dataset(REGRESSION + "auto-mpg.csv", "MPG", TargetType.Regression)

    ds.numbers(["HP"])
    ds.handle_na()
    ds.normalize(excepts=["Cylinders","Year","Origin"])
    return (1000, LinearRegression(ds, learning_rate=0.0001))

def automobile() -> tuple[int, MLAlgorithm]:
    ds = Dataset(REGRESSION + "automobile.csv", "symboling", TargetType.Regression)

    attributes_to_modify = ["fuel-system", "engine-type", "drive-wheels", "body-style", "make", "engine-location", "aspiration", "fuel-type", "num-of-cylinders", "num-of-doors"]
    ds.factorize(attributes_to_modify)
    ds.numbers(["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm", "price"])
    ds.handle_na()
    ds.normalize(excepts=attributes_to_modify)
    return (1000, LinearRegression(ds, learning_rate=0.004))

def power_plant() -> tuple[int, MLAlgorithm]:
    ds = Dataset(REGRESSION + "power-plant.csv", "energy-output", TargetType.Regression)
    ds.normalize()
    return (80, LinearRegression(ds, learning_rate=0.1))


def electrical_grid() -> tuple[int, MLAlgorithm]:
    ds = Dataset(CLASSIFICATION + "electrical_grid.csv", "stabf", TargetType.Classification)
    ds.factorize(["stabf"])
    ds.normalize()
    return (1000, LogisticRegression(ds, learning_rate=0.08))

def frogs() -> tuple[int, MLAlgorithm]:
    ds = Dataset(CLASSIFICATION + "frogs.csv", "Species", TargetType.MultiClassification)
    ds.remove(["Family", "Genus", "RecordID"])
    ds.factorize(["Species"])
    return (1000, MultiLayerPerceptron(ds, learning_rate=0.08))




def learn_dataset(function:Callable[..., tuple[int, MLAlgorithm]], epochs:int=10000, verbose=True)-> MLAlgorithm:
    skip, ml = function()
    ml.learn(epochs, verbose=verbose)

    err_tests = ml.test_loss()
    err_valid = ml.validation_loss()
    err_learn = ml.learning_loss()
    print(f"Loss value: tests={err_tests:1.5f}, valid={err_valid:1.5f}, learn={err_learn:1.5f}")

    ml.plot(skip=skip)
    return ml

if __name__ == "__main__":
    ml = learn_dataset(electrical_grid)
    print(ml.accuracy(ml.testset))
