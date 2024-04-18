from learning.data import Dataset
from learning.supervised import LinearRegression
from learning.ml import MLAlgorithm
from plot import Plot

def auto_mpg() -> MLAlgorithm:
    df = Dataset("datasets\\auto-mpg.csv", "MPG")

    df.to_numbers(["HP"])
    df.handle_na()
    df.regularize(excepts=["Cylinders","Year","Origin"])

    return LinearRegression(df, learning_rate=0.0001)

def automobile() -> MLAlgorithm:
    df = Dataset("datasets\\regression\\automobile.csv", "symboling")

    attributes_to_modify = ["fuel-system", "engine-type", "drive-wheels", "body-style", "make", "engine-location", "aspiration", "fuel-type", "num-of-cylinders", "num-of-doors"]
    df.factorize(attributes_to_modify)
    df.to_numbers()
    df.handle_na()
    df.regularize(excepts=attributes_to_modify)

    return LinearRegression(df, learning_rate=0.001)




epoch = 50000
skip = 1000
lr = automobile()

train, test = lr.learn(epoch)

plot = Plot("Error", "Time", "Mean Error")
plot.line("training", "red", data=train[skip:])
plot.line("test", "blue", data=test[skip:])

"""
for _ in range(0, epoch):
    train_err = lr.learning_step()
    test_err = lr.test_error()

    plot.update("training", train_err)
    plot.update("test", test_err)
    plot.update_limits()
"""

plot.wait()


