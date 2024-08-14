import numpy as np
import sklearn
import sklearn.cluster
import sklearn.linear_model
import sklearn.model_selection
import sklearn.neural_network

from typing import Any
from learning.ml import MLAlgorithm
from learning.data import Dataset, TargetType
from learning.supervised import LinearRegression, LogisticRegression, MultiLayerPerceptron
from learning.unsupervised import KMeans

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
    size = [18, 15, 12, 10, 8]
    return (ds, MultiLayerPerceptron(ds, size), sklearn.neural_network.MLPClassifier(size, 'relu'))

def iris() -> tuple[Dataset, MLAlgorithm, Any]:
    ds = Dataset(CLASSIFICATION + "iris.csv", "Class", TargetType.MultiClassification)
    ds.factorize(["Class"])
    ds.normalize()
    size = [4, 3]
    return (ds, MultiLayerPerceptron(ds, size), sklearn.neural_network.MLPClassifier(size, 'relu'))

# ********************
# MultiLayerPerceptron
# ********************

def frogs_no_target() -> tuple[Dataset, MLAlgorithm, Any]:
    ds = Dataset(CLASSIFICATION + "frogs.csv", "Species", TargetType.NoTarget)
    ds.remove(["Family", "Genus", "RecordID", "Species"])
    clusters = 10
    return (ds, KMeans(ds, clusters), sklearn.cluster.KMeans(clusters))

def iris_no_target() -> tuple[Dataset, MLAlgorithm, Any]:
    ds = Dataset(CLASSIFICATION + "iris.csv", "Class", TargetType.NoTarget)
    ds.remove(["Class"])
    ds.normalize()
    clusters = 3
    return (ds, KMeans(ds, clusters), sklearn.cluster.KMeans(clusters))

# ********************
# Main & random
# ********************

if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf, formatter={'float': '{:>10.5f}'.format})
    rand = np.random.randint(0, 4294967295)
    #rand = 1997847910  # LiR for power_plant
    #rand = 347617386   # LoR for electrical_grid
    #rand = 1793295160  # MLP for iris
    #rand = 2914000170  # MLP for frogs
    #rand = 885416001   # KMe for frogs_no_target

    np.random.seed(rand)
    print(f"Using seed: {rand}")

    ds, ml, sk = frogs()

    epochs, _, _ = ml.learn(1000, verbose=True)
    ml.display_results()

    np.random.seed(rand)
    learn, test, valid = ds.get_dataset()
    sk.set_params(max_iter=epochs)
    sk.fit(learn.x, learn.y)
    print(f"Sklearn    : {abs(sk.score(test.x, test.y)):0.5f}")
    print("========================")

    ml.plot()
