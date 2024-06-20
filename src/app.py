import random
from typing import Any
import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.model_selection
import sklearn.neural_network
from learning.data import Dataset, TargetType
from learning.supervised import LinearRegression, LogisticRegression, MultiLayerPerceptron
from learning.ml import MLAlgorithm

DATASET = "datasets/"
REGRESSION = DATASET + "regression/"
CLASSIFICATION = DATASET + "classification/"

# ********************
# Linear Regression
# ********************

def auto_mpg() -> tuple[Dataset, MLAlgorithm, Any]:
    ds = Dataset(REGRESSION + "auto-mpg.csv", "MPG", TargetType.Regression)

    ds.numbers(["HP"])
    ds.handle_na()
    ds.normalize(excepts=["Cylinders","Year","Origin"])
    return (ds, LinearRegression(ds, learning_rate=0.0001), sklearn.linear_model.LinearRegression())

def automobile() -> tuple[Dataset, MLAlgorithm, Any]:
    ds = Dataset(REGRESSION + "automobile.csv", "symboling", TargetType.Regression)

    attributes_to_modify = ["fuel-system", "engine-type", "drive-wheels", "body-style", "make", "engine-location", "aspiration", "fuel-type", "num-of-cylinders", "num-of-doors"]
    ds.factorize(attributes_to_modify)
    ds.numbers(["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm", "price"])
    ds.handle_na()
    ds.normalize(excepts=attributes_to_modify)
    return (ds, LinearRegression(ds, learning_rate=0.004), sklearn.linear_model.LinearRegression())

def power_plant() -> tuple[Dataset, MLAlgorithm, Any]:
    ds = Dataset(REGRESSION + "power-plant.csv", "energy-output", TargetType.Regression)
    ds.normalize()
    return (ds, LinearRegression(ds, learning_rate=0.1), sklearn.linear_model.LinearRegression())

# ********************
# Logistic Regression
# ********************

def electrical_grid() -> tuple[Dataset, MLAlgorithm, Any]:
    ds = Dataset(CLASSIFICATION + "electrical_grid.csv", "stabf", TargetType.Classification)
    ds.factorize(["stabf"])
    ds.normalize()
    return (ds, LogisticRegression(ds, learning_rate=100), sklearn.linear_model.LogisticRegression())

def heart() -> tuple[Dataset, MLAlgorithm, Any]:
    ds = Dataset(CLASSIFICATION + "heart.csv", "Disease", TargetType.Classification)
    attributes_to_modify = ["Disease", "Sex", "ChestPainType"]
    ds.factorize(attributes_to_modify)
    ds.normalize(excepts=attributes_to_modify)
    return (ds, LogisticRegression(ds, learning_rate=0.01), sklearn.linear_model.LogisticRegression())

# ********************
# MultiLayerPerceptron
# ********************

def frogs() -> tuple[Dataset, MLAlgorithm, Any]:
    ds = Dataset(CLASSIFICATION + "frogs.csv", "Species", TargetType.MultiClassification)
    ds.remove(["Family", "Genus", "RecordID"])
    ds.factorize(["Species"])
    return (ds, MultiLayerPerceptron(ds, [4, 3]), sklearn.neural_network.MLPClassifier([4, 3], 'relu'))

def iris() -> tuple[Dataset, MLAlgorithm, Any]:
    ds = Dataset(CLASSIFICATION + "iris.csv", "Class", TargetType.MultiClassification)
    ds.factorize(["Class"])
    ds.normalize()
    return (ds, MultiLayerPerceptron(ds, [4, 3]), sklearn.neural_network.MLPClassifier([4, 3], 'relu'))

# ********************
# Main & random
# ********************

if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf, formatter={'float': '{:>10.5f}'.format})
    rand = random.randint(0, 4294967295)
    np.random.seed(rand)
    print(f"Using seed: {rand}")

    ds, ml, sk = electrical_grid()
    ml.learn(10000, verbose=True)
    ml.display_results()

    np.random.seed(rand)
    learn, test, valid = ds.get_dataset()
    sk.fit(learn.x, learn.y)
    print(f"Sklearn    : {sk.score(test.x, test.y):0.5f}")
    print("========================")

    ml.plot()

# migliori parametri trovati per electrical_grid
# temp = np.array([-48.28601, 0.00429, 0.07933, 0.02144, -0.04225, 0.36898, 0.24723, 0.36445, 0.21437, 0.29666, 0.22532, 0.38619, 0.24171, -113.65430])
# ml._set_parameters(temp)
