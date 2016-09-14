from sklearn import datasets
import random
import numpy as np
from hardCoded import HardCoded

def check_accuracy(predicted, actual):
    if predicted == actual:
        return 1
    else:
        return 0

iris = datasets.load_iris()

# Shuffle the data and target together
shuffled = list(zip(iris.data, iris.target))

random.shuffle(shuffled)

# Separate out data and target
dataList, targetList = zip(*shuffled)
dataArray = np.asarray(dataList)
targetArray = np.asarray(targetList)

# Split arrays by length * 70%
dataSplit = int(len(dataArray) * 0.7)
print(dataSplit)

# Split data
trainingData = dataArray[:dataSplit]
testData = dataArray[dataSplit:]

# Split target
trainingTarget = targetArray[:dataSplit]
testTarget = targetArray[dataSplit:]

# Test with our classifier
tester = HardCoded()
tester.train(trainingData, trainingTarget)

result = tester.predict(testData)

# Show result
print(result)

numRight = 0
for predicted, actual in zip(result, testTarget):
    numRight += check_accuracy(predicted, actual)

print("Accuracy: %0.2f" % (numRight / len(testTarget)))
