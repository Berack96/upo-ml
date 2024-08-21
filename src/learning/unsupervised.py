import math as math
import numpy as np

from learning.ml import MLAlgorithm
from learning.data import Dataset, Data

class KMeans(MLAlgorithm):
    def __init__(self, dataset: Dataset, clusters:int) -> None:
        super().__init__(dataset)
        dimensions = self._learnset.x.shape[1]
        self.total = clusters
        self.centroids = np.random.rand(clusters, dimensions)

    def _h0(self, x:np.ndarray) -> np.ndarray:
        diff = x[:, np.newaxis] - self.centroids
        distances = np.linalg.norm(diff, axis=2)
        return np.argmin(distances, axis=1)

    def _predict_loss(self, dataset:Data) -> float:
        assignments = self._h0(dataset.x)
        loss = 0.0

        for k in range(self.total):
            assigned_points = dataset.x[assignments == k]
            if len(assigned_points) > 0:
                diff = assigned_points - self.centroids[k]
                loss += np.sum(np.linalg.norm(diff, axis=1) ** 2)
        return loss

    def _learning_step(self) -> float:
        assignments = self._h0(self._learnset.x)
        centroids = []

        for k in range(self.total):
            assigned_points = self._learnset.x[assignments == k]

            if len(assigned_points) > 0:
                mean = np.mean(assigned_points, axis=0)
                centroids.append(mean)
            else:
                self.total -= 1

        self.centroids = np.array(centroids)
        return self._predict_loss(self._learnset)


    def _get_parameters(self):
        return self.centroids.copy()
    def _set_parameters(self, parameters):
        self.centroids = parameters
