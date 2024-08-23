import numpy as np
import sklearn
import sklearn.cluster
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.neural_network

from typing import Any
from learning.functions import print_metrics
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
    ds.standardize(excepts=["Cylinders","Year","Origin"])
    return (ds, LinearRegression(ds, learning_rate=0.0001), sklearn.linear_model.SGDRegressor())

def automobile() -> tuple[Dataset, MLAlgorithm, Any]:
    ds = Dataset(REGRESSION + "automobile.csv", "symboling", TargetType.Regression)

    attributes_to_modify = ["fuel-system", "engine-type", "drive-wheels", "body-style", "make", "engine-location", "aspiration", "fuel-type", "num-of-cylinders", "num-of-doors"]
    ds.factorize(attributes_to_modify)
    ds.numbers(["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm", "price"])
    ds.handle_na()
    ds.standardize(excepts=attributes_to_modify)
    return (ds, LinearRegression(ds, learning_rate=0.003), sklearn.linear_model.SGDRegressor())

def power_plant() -> tuple[Dataset, MLAlgorithm, Any]:
    ds = Dataset(REGRESSION + "power-plant.csv", "energy-output", TargetType.Regression)
    ds.standardize(excepts=None)
    return (ds, LinearRegression(ds, learning_rate=0.1), sklearn.linear_model.SGDRegressor())

# ********************
# Logistic Regression
# ********************

def electrical_grid() -> tuple[Dataset, MLAlgorithm, Any]:
    ds = Dataset(CLASSIFICATION + "electrical_grid.csv", "stabf", TargetType.Classification)
    ds.factorize(["stabf"])
    ds.standardize()
    return (ds, LogisticRegression(ds, learning_rate=100), sklearn.linear_model.LogisticRegression())

def heart() -> tuple[Dataset, MLAlgorithm, Any]:
    ds = Dataset(CLASSIFICATION + "heart.csv", "Disease", TargetType.Classification)
    attributes_to_modify = ["Disease", "Sex", "ChestPainType"]
    ds.factorize(attributes_to_modify)
    ds.standardize(excepts=attributes_to_modify)
    return (ds, LogisticRegression(ds, learning_rate=0.01), sklearn.linear_model.LogisticRegression())

# ********************
# MultiLayerPerceptron
# ********************

def electrical_grid_mlp() -> tuple[Dataset, MLAlgorithm, Any]:
    ds = Dataset(CLASSIFICATION + "electrical_grid.csv", "stabf", TargetType.MultiClassification)
    ds.factorize(["stabf"])
    ds.standardize()
    size = [4, 3]
    return (ds, MultiLayerPerceptron(ds, size, 0.05), sklearn.neural_network.MLPClassifier(size, 'relu'))

def frogs() -> tuple[Dataset, MLAlgorithm, Any]:
    ds = Dataset(CLASSIFICATION + "frogs.csv", "Family", TargetType.MultiClassification)
    ds.remove(["Species", "Genus", "RecordID"])
    ds.factorize(["Family"])
    ds.standardize()
    size = [18, 12, 8]
    return (ds, MultiLayerPerceptron(ds, size, 0.02), sklearn.neural_network.MLPClassifier(size, 'relu'))

def iris() -> tuple[Dataset, MLAlgorithm, Any]:
    ds = Dataset(CLASSIFICATION + "iris.csv", "Class", TargetType.MultiClassification)
    ds.factorize(["Class"])
    ds.standardize()
    size = [4, 3]
    return (ds, MultiLayerPerceptron(ds, size), sklearn.neural_network.MLPClassifier(size, 'relu'))

# ********************
# MultiLayerPerceptron
# ********************

def frogs_no_target() -> tuple[Dataset, MLAlgorithm, Any]:
    ds = Dataset(CLASSIFICATION + "frogs.csv", "Family", TargetType.NoTarget)
    ds.remove(["Family", "Genus", "RecordID", "Species"])
    clusters = 4
    return (ds, KMeans(ds, clusters), sklearn.cluster.KMeans(clusters))

def iris_no_target() -> tuple[Dataset, MLAlgorithm, Any]:
    ds = Dataset(CLASSIFICATION + "iris.csv", "Class", TargetType.NoTarget)
    ds.remove(["Class"])
    clusters = 3
    return (ds, KMeans(ds, clusters), sklearn.cluster.KMeans(clusters))

# ********************
# Main & random
# ********************

if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf, formatter={'float': '{:>10.5f}'.format})
    rand = np.random.randint(0, (1 << 31) - 1)
    #rand = 2205910060  # LiR for power_plant
    #rand = 347617386   # LoR for electrical_grid
    #rand = 834535453   # LoR for heart
    #rand = 1793295160  # MLP for iris
    #rand = 772284034   # MLP for frogs
    #rand = 1038336550  # KMe for frogs_no_target

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
    print_metrics(ml._target_type, test, sk.predict(test.x))

    ml.plot()
