class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None


class DecisionTree(object):
    def __init__(self):
        self.Tree = Tree()
        self.data = None

    def train(self, attributes, features, data):
        pass

    def predict(self):
        pass
