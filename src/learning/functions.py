import numpy as np

from learning.data import ConfusionMatrix, Data, Dataset, TargetType
from sklearn.metrics import silhouette_score, r2_score

NOT_ZERO = 1e-15
LEAKY_RELU = 0.2


# **********
# For NN
# **********

def relu(x:np.ndarray) -> np.ndarray:
    return np.where(x < 0, 0, x)
def relu_derivative(x:np.ndarray) -> np.ndarray:
    return np.where(x < 0, 0, 1)

def lrelu(x:np.ndarray) -> np.ndarray:
    return np.where(x < 0, LEAKY_RELU * x, x)
def lrelu_derivative(x:np.ndarray) -> np.ndarray:
    return np.where(x < 0, LEAKY_RELU, 1)

def softmax(x:np.ndarray) -> np.ndarray:
    axis = 1 if len(x.shape) != 1 else 0
    x = x - np.max(x, axis=axis, keepdims=True) # for overflow
    exp_x = np.exp(x)
    sum_x = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_x
def softmax_derivative(h0:np.ndarray, y:np.ndarray) -> np.ndarray:
    return h0 - y

# **********
# For loss
# **********

def square_loss(h0:np.ndarray, y:np.ndarray) -> float:
    return np.mean((h0 - y) ** 2) / 2

def log_loss(h0:np.ndarray, y:np.ndarray) -> float:
    return np.mean(- y*np.log(h0 + NOT_ZERO) - (1-y)*np.log(1-h0 + NOT_ZERO))

def cross_entropy_loss(h0:np.ndarray, y:np.ndarray) -> float:
    return -np.mean(np.sum(y*np.log(h0 + NOT_ZERO), axis=1)) # mean is not "correct", but useful for comparing models


# **********
# Randoms
# **********

def with_bias(x:np.ndarray) -> np.ndarray:
    shape = (x.shape[0], 1) if len(x.shape) != 1 else (1,)
    ones = np.ones(shape)
    return np.hstack([ones, x])

def print_metrics(target:TargetType, dataset:Data, h0:np.ndarray) -> None:
    if target == TargetType.Regression:
        print(f"R^2        : {r2_score(dataset.y, h0):0.5f}")
        print(f"Pearson    : {np.corrcoef(dataset.y, h0)[0, 1]:0.5f}")
    elif target != TargetType.NoTarget:
        if h0.ndim == 1: h0 = np.where(h0 > 0.5, 1, 0)
        ConfusionMatrix(dataset.y, h0).print()
    else:
        print(f"Silhouette : {silhouette_score(dataset.x, h0):0.5f}")
    print("========================")

def print_silhouette_weka(ds:Dataset, file_weka:str):
    test, _, _, _ = ds.get_dataset()[2].as_tuple()
    test = np.round(test, 6)

    weka = Dataset(file_weka, "", TargetType.NoTarget)
    weka.factorize(["cluster"])

    weka, _, _, _ = weka.get_dataset(test_frac=0, valid_frac=0)[0].as_tuple()
    weka_x, weka_y = weka[:, :-1], weka[:, -1:]

    bau = [np.where((weka_x == x).all(axis=1))[0][0] for x in test]
    weka_x, weka_y = weka_x[bau], weka_y[bau].ravel()

    score = silhouette_score(weka_x, weka_y)
    print(score)
