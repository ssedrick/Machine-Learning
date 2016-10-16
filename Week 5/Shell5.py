import pandas as pd
import random
import numpy as np
from sklearn import datasets
from neuralNetwork import NeuralNet


class SourceData:
    def __init__(self):
        self.data = []
        self.target = []


def check_accuracy(guess, real):
    if guess == real:
        return 1
    else:
        return 0


def get_split_percentage():
    percent = -1
    while not(0 < percent < 100):
        percent = float(input("What percentage of the dataset should be used for training? "))
        if not(0 < percent < 100):
            print("Sorry, I need a number between 0 and 100. It can have a decimal.")

    return percent / 100


def load_car_dataset():
    df = pd.read_csv('cars.csv')
    car = df.values
    data = SourceData()
    data.data, data.target = car[:, :6], car[:, 6]
    return data


def load_votes_csv():
    df = pd.read_csv('votes.csv', header=None)
    votes = df.values
    data = SourceData()
    data.target, data.data = votes[:, 0], votes[:, 1:]
    return data


def load_loans_csv():
    df = pd.read_csv('loans.csv')
    loans = df.values
    data = SourceData()
    data.data, data.target = loans[:, :4], loans[:, 4]
    return data


def load_lenses_csv():
    df = pd.read_csv('lenses.csv', header=None)
    votes = df.values
    data = SourceData()
    data.target, data.data = votes[:, 4], votes[:, :4]
    return data


def load_iris():
    iris = datasets.load_iris()
    targets = []
    for i in range(len(iris.target)):
        if iris.target[i] == 0:
            targets.append([1, 0, 0])
        elif iris.target[i] == 1:
            targets.append([0, 1, 0])
        else:
            targets.append([0, 0, 1])
    data = SourceData()
    data.data, data.target = np.asarray(iris.data), np.asarray(targets)
    return data


def get_dataset():
    i = input("Which dataset should I load? [1] Iris, [2] Cars, [3] Breast Cancer, [4] Votes, [5] Loans, [6] Lenses ")
    if i == '1' or i.lower == 'iris':
        return load_iris()

    if i == '2' or i.lower == 'cars':
        return load_car_dataset()

    if i == '3' or i.lower == 'breast cancer':
        return datasets.load_breast_cancer()

    if i == '4' or i.lower == 'votes':
        return load_votes_csv()

    if i == '5' or i.lower == 'loans':
        return load_loans_csv()

    if i == '6' or i.lower == 'lenses':
        return load_lenses_csv()


def main():
    source = get_dataset()

    # Shuffle the data and target together
    shuffled = list(zip(source.data, source.target))

    random.shuffle(shuffled)

    # Separate out data and target
    data_list, target_list = list(zip(*shuffled))

    # Split arrays by length * percentage
    data_split = int(len(data_list) * get_split_percentage())

    # Split data
    training_data = data_list[:data_split]
    test_data = data_list[data_split:]

    # Split target
    training_target = target_list[:data_split]
    test_target = target_list[data_split:]

    print("Building the tree...")
    # Test with our classifier
    tester = NeuralNet([3], num_inputs=4)
    print(tester)
    tester.train(data=training_data, targets=training_target)
    print(tester)

    result = tester.predict(test_data)

    # Count the number right
    num_right = 0
    for predicted, actual in zip(result, test_target):
        num_right += 1 if tester.check_output(predicted, actual) else 0

    # Show Accuracy
    print("Accuracy: %2.2f%%" % (num_right * 100 / len(test_target)))

if __name__ == "__main__":
    main()
