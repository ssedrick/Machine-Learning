import pandas as pd
import random
import numpy as np
from sklearn import datasets
from decisionTree import DecisionTree


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
    df = df.replace('vhigh', 4).replace('high', 3).replace('med', 2).replace('low', 1)
    df = df.replace('5more', 6).replace('more', 5)
    df = df.replace('small', 1).replace('med', 2).replace('big', 3)
    df = df.replace('unacc', 1).replace('acc', 2).replace('good', 3).replace('vgood', 4)
    car = df.values
    data = SourceData()
    data.data, data.target = car[:, :6], car[:, 6]
    data.data, data.target = data.data.astype(int), data.target.astype(int)
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


def get_dataset():
    return load_loans_csv()
    # return load_votes_csv()
    """
    i = input("Which dataset should I load? [1] Iris, [2] Cars, [3] Breast Cancer ")
    if i == '1' or i.lower == 'iris':
        return datasets.load_iris()

    if i == '2' or i.lower == 'cars':
        return load_car_dataset()

    if i == '3' or i.lower in 'breast cancer':
        return datasets.load_breast_cancer()
    """


def main():
    source = get_dataset()

    # Shuffle the data and target together
    shuffled = list(zip(source.data, source.target))

    random.shuffle(shuffled)

    # Separate out data and target
    data_list, target_list = zip(*shuffled)
    data_array = np.asarray(data_list)
    target_array = np.asarray(target_list)

    # Split arrays by length * percentage
    data_split = int(len(data_array) * get_split_percentage())

    # Split data
    training_data = data_array[:data_split]
    test_data = data_array[data_split:]

    # Split target
    training_target = target_array[:data_split]
    test_target = target_array[data_split:]

    # Test with our classifier
    tester = DecisionTree()
    tester.train(training_data, training_target)

    """
    result = tester.predict(test_data)

    # Count the number right
    num_right = 0
    for predicted, actual in zip(result, test_target):
        num_right += check_accuracy(predicted, actual)

    # Show Accuracy
    print("Accuracy: %2.2f%%" % (num_right * 100 / len(test_target)))
    """

if __name__ == "__main__":
    main()
