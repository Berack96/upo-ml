import numpy as np

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

def pearson(h0:np.ndarray, y:np.ndarray) -> float:
    diff1 = h0 - h0.mean()
    diff2 = y - y.mean()
    num = np.sum(diff1 * diff2)
    den = np.sqrt(np.sum(diff1**2)) * np.sqrt(np.sum(diff2**2))
    return num / den

def r_squared(h0:np.ndarray, y:np.ndarray) -> float:
    y_mean = np.mean(y)
    ss_resid = np.sum((y - h0) ** 2)
    ss_total = np.sum((y - y_mean) ** 2)
    return 1 - (ss_resid / ss_total)

def with_bias(x:np.ndarray) -> np.ndarray:
    shape = (x.shape[0], 1) if len(x.shape) != 1 else (1,)
    ones = np.ones(shape)
    return np.hstack([ones, x])
