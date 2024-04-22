from learning.data import Dataset
from learning.supervised import LinearRegression
from learning.ml import MLRegression
from typing import Callable

def auto_mpg() -> tuple[int, MLRegression]:
    df = Dataset("datasets\\auto-mpg.csv", "MPG")

    df.to_numbers(["HP"])
    df.handle_na()
    df.regularize(excepts=["Cylinders","Year","Origin"])
    return (1000, LinearRegression(df, learning_rate=0.0001))

def automobile() -> tuple[int, MLRegression]:
    df = Dataset("datasets\\regression\\automobile.csv", "symboling")

    attributes_to_modify = ["fuel-system", "engine-type", "drive-wheels", "body-style", "make", "engine-location", "aspiration", "fuel-type", "num-of-cylinders", "num-of-doors"]
    df.factorize(attributes_to_modify)
    df.to_numbers(["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm", "price"])
    df.handle_na()
    df.regularize(excepts=attributes_to_modify)
    return (1000, LinearRegression(df, learning_rate=0.004))

def power_plant() -> tuple[int, MLRegression]:
    df = Dataset("datasets\\regression\\power-plant.csv", "energy-output")
    df.regularize()
    return (80, LinearRegression(df, learning_rate=0.1))



def learn_dataset(function:Callable[..., tuple[int, MLRegression]], epochs:int=100000, verbose=True)-> None:
    skip, ml = function()
    ml.learn(epochs, verbose=verbose)

    err_tests = ml.test_loss()
    err_valid = ml.validation_loss()
    err_learn = ml.learning_loss()
    print(f"Loss value: tests={err_tests:1.5f}, valid={err_valid:1.5f}, learn={err_learn:1.5f}")

    ml.plot(skip=skip)



if __name__ == "__main__":
    learn_dataset(auto_mpg)
