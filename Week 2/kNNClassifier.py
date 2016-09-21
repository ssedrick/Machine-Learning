import numpy as np


class KNNClassifier(object):
    def __init__(self, neighbors):
        self.k = neighbors
        self.data = []
        self.targets = []

    def train(self, data, targets):
        self.data = data
        self.targets = targets

    def predict(self, inputs):
        n_inputs = np.shape(inputs)[0]
        closest = np.zeros(n_inputs)

        for n in range(n_inputs):
            # Compute Distances
            distances = np.sum((self.data-inputs[n])**2, axis=1)

            indices = np.argsort(distances, axis=0)

            classes = np.unique(self.targets[indices[:self.k]])
            if len(classes) == 1:
                closest[n] = np.unique(classes)
            else:
                counts = np.zeros(max(classes) + 1)
                for i in range(self.k):
                    counts[self.targets[indices[i]]] += 1
                closest[n] = np.max(counts)

        return closest
