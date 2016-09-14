class HardCoded:

    def __init__(self):
        self.predictions = []

    def train(self, trainingdata, trainingtarget):
        return

    def predict(self, testdata):
        self.predictions = [1 for item in testdata]
        return self.predictions
