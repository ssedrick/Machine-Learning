import numpy as np
from sklearn.datasets.base import Bunch


class Tree(object):
    def __init__(self):
        self.data = None
        self.branches = None


class DecisionTree(object):
    def __init__(self):
        self.Tree = Tree()
        self.data = None
        self.targets = None
        self.classes = None
        self.attributes = Bunch()

    def extract_attributes(self, data):
        for i in range(len(data[0])):
            self.attributes[i] = np.unique(data[:, i])

    def build_tree(self, node, data, targets, features, attributes):
        if len(np.unique(targets)) == 1:
            node.data = targets[0]
            print("Trimmed down to just one target! ", node.data)
            return node
        elif len(features) == 1:
            node.data = features[0]
            print("Run out of features: ", node.data)
            return node
        else:
            # We still have features to look at
            entropies = self.calc_entropies(data, targets, features, attributes)
            column, best = -1, 0
            for col in entropies:
                if entropies[col] > best:
                    best = entropies[col]
                    column = col

            features.remove(column)
            node.data = column
            node.branches = Bunch()
            print(node.data)
            for attribute in attributes[col]:
                node.branches[attribute] = Tree()
                self.build_tree(node.branches[attribute], data, targets, features, attributes)

    def entropy(self, data, targets, feature, attributes):
        frequencies = Bunch()
        data_partition_size = Bunch()
        for attribute in attributes:
            frequencies[attribute] = Bunch()
            data_partition_size[attribute] = 0
            for cls in self.classes:
                frequencies[attribute][cls] = 0

        for i in range(len(data)):
            frequencies[data[i, feature]][targets[i]] += 1
            data_partition_size[data[i, feature]] += 1

        for attribute in attributes:
            for cls in self.classes:
                frequencies[attribute][cls] /= data_partition_size[attribute]

        entropy = 0
        size = len(data)
        for attrib in frequencies:
            attrib_entropy = 0
            for cls in frequencies[attrib]:
                if frequencies[attrib][cls] != 0.0:
                    attrib_entropy -= frequencies[attrib][cls] * np.log2(frequencies[attrib][cls])
            entropy += data_partition_size[attrib] / size * attrib_entropy

        print(entropy)
        return entropy

    def calc_entropies(self, data, targets, features, attributes):
        entropies = Bunch()
        for feature in features:
            entropies[feature] = self.entropy(data, targets, feature, attributes[feature])
        return entropies

    def train(self, data, targets):
        self.data = data
        self.targets = targets
        self.classes = np.unique(targets)
        self.extract_attributes(data)
        features = list(range(0, len(self.data[0])))
        self.build_tree(self.Tree, self.data, self.targets, features, self.attributes)

    def predict(self):
        pass
