from learning.data import Dataset
from learning.supervised import LinearRegression
from learning.ml import MLRegression

def auto_mpg() -> tuple[int, int, MLRegression]:
    df = Dataset("datasets\\auto-mpg.csv", "MPG")

    df.to_numbers(["HP"])
    df.handle_na()
    df.regularize(excepts=["Cylinders","Year","Origin"])
    return (5000, 1000, LinearRegression(df, learning_rate=0.0001))

def automobile() -> tuple[int, int, MLRegression]:
    df = Dataset("datasets\\regression\\automobile.csv", "symboling")

    attributes_to_modify = ["fuel-system", "engine-type", "drive-wheels", "body-style", "make", "engine-location", "aspiration", "fuel-type", "num-of-cylinders", "num-of-doors"]
    df.factorize(attributes_to_modify)
    df.to_numbers(["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm", "price"])
    df.handle_na()
    df.regularize(excepts=attributes_to_modify)
    return (5000, 1000, LinearRegression(df, learning_rate=0.002))

def power_plant() -> tuple[int, int, MLRegression]:
    df = Dataset("datasets\\regression\\power-plant.csv", "energy-output")
    df.regularize()
    return (1000, 80, LinearRegression(df, learning_rate=0.1))

epoch, skip, ml = automobile()
ml.learn(epoch)
ml.plot(skip=skip)

"""
for _ in range(0, epoch):
    train_err = lr.learning_step()
    test_err = lr.test_error()

    plot.update("training", train_err)
    plot.update("test", test_err)
    plot.update_limits()
"""
