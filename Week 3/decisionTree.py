import numpy as np
import pandas
from sklearn.datasets.base import Bunch


class Node(object):
    def __init__(self):
        self.data = None
        self.branches = None

    def __str__(self, level=0):
        ret = "|---" * level + repr(self.data) + "\n"
        if self.branches is not None:
            for branch in self.branches:
                ret += self.branches[branch].__str__(level + 1)
        return ret

    def __repr__(self):
        return '<tree node>'


class DecisionTree(object):
    def __init__(self):
        self.Tree = Node()
        self.df = pandas.DataFrame()
        self.classes = None
        self.attributes = Bunch()

    def extract_attributes(self, data):
        for i in range(len(data[0])):
            self.attributes[i] = np.unique(data[:, i])

    def split_data(self, df, features, feature, attribute):
        split_data_frame = df[df[feature] == attribute]
        del split_data_frame[feature]
        features.remove(feature)
        return split_data_frame, features

    def pick_best(self, entropies):
        column, best = -1, 1.1
        for col in entropies:
            if entropies[col] < best:
                best = entropies[col]
                column = col
        return column

    def pick_leaf(self, df, targets):
        target_frequencies = Bunch()
        for target in targets:
            target_frequencies[target] = df[df["targets"] == target].size
        best_target, best_count = '', -1
        for target in target_frequencies:
            if target_frequencies[target] > best_count:
                best_count = target_frequencies[target]
                best_target = target
        return best_target

    def build_tree(self, node, df, features, attributes):
        if len(np.unique(df["targets"])) == 1:
            node.data = np.unique(df["targets"])[0]
            # print("Trimmed down to just one target! ", node.data)
        elif len(features) == 1:
            node.data = self.pick_leaf(df, df["targets"])
            # print("Run out of features: ", node.data)
        else:
            # We still have features to look at
            entropies = self.calc_entropies(df, features, attributes)
            best = self.pick_best(entropies)
            node.data = best
            node.branches = Bunch()
            for attribute in attributes[best]:
                split_df, split_features = self.split_data(df[:], features[:], best, attribute)
                node.branches[attribute] = Node()
                if not split_df.empty:
                    self.build_tree(node.branches[attribute], split_df, split_features, attributes)
                else:
                    node.branches[attribute].data = 'undecided'

    def entropy(self, df, feature, attributes):
        """
        Calculate the entropy of a given feature based on the data and attributes
        :param df: a data frame of datapoints
        :param feature: A column in each row of data that we are looking at
        :param attributes: A Bunch of all possible attributes for a given row
        :return:
        """
        frequencies = Bunch()
        data_partition_size = Bunch()

        """ Set up for calculating Frequencies"""
        for attribute in attributes:
            frequencies[attribute] = Bunch()
            data_partition_size[attribute] = 0
            for cls in self.classes:
                frequencies[attribute][cls] = 0

        # print("Calculating Frequencies of feature: ", feature)
        """ Loop through data to get the frequencies of the attribute and size of data for each category"""
        for row in df.iterrows():
            frequencies[row[1][feature]][row[1]["targets"]] += 1
            data_partition_size[row[1][feature]] += 1

        # print("Frequencies calculated: ", frequencies)
        """ Turn frequencies into percentages """
        for attribute in attributes:
            for cls in self.classes:
                if data_partition_size[attribute] != 0:
                    frequencies[attribute][cls] /= data_partition_size[attribute]
                else:
                    frequencies[attribute][cls] = 0.0

        """ Calculate entropy from frequencies """
        entropy = 1
        size = len(df)
        if size != 0:
            for attrib in frequencies:
                attrib_entropy = 0
                for cls in frequencies[attrib]:
                    if frequencies[attrib][cls] != 0.0:
                        attrib_entropy -= frequencies[attrib][cls] * np.log2(frequencies[attrib][cls])
                entropy -= data_partition_size[attrib] / size * attrib_entropy

        # print(entropy)
        return entropy

    def calc_entropies(self, df, features, attributes):
        entropies = Bunch()
        for feature in features:
            entropies[feature] = self.entropy(df, feature, attributes[feature])
        return entropies

    def train(self, data, targets):
        features = list(range(0, len(data[0])))
        self.df = pandas.DataFrame(data=data, columns=features)
        self.df['targets'] = targets
        self.classes = np.unique(targets)
        self.extract_attributes(data)
        self.build_tree(self.Tree, self.df, features, self.attributes)

    def traverse(self, node, item):
        if node.branches is not None:
            return self.traverse(node.branches[item[node.data]], item)
        else:
            return node.data

    def predict(self, data):
        return self.traverse(self.Tree, data)

    def print(self):
        print(self.Tree)
