import pandas as pd
import random
from sklearn import datasets
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.datasets.base import Bunch
from copy import deepcopy
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


def get_learning_rate(default):
    rate = input("How fast do you want the network to learn? ")
    if rate is not "":
        return float(rate)
    else:
        return default


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


def load_pima_csv():
    df = pd.read_csv('pima.csv')
    pima = df.values
    data = SourceData()
    data.data, data.target = pima[:, :8], pima[:, 8]
    data.data = preprocessing.normalize(data.data)
    target = []
    data_list = []
    for i in range(len(data.target)):
        if data.target[i] == 0:
            target.append([0, 0, 1, 1])
        else:
            target.append([1, 1, 0, 0])
        data_list.append(list(data.data[i]))
    data.target = target
    data.data = data_list
    return data


def load_iris():
    iris = datasets.load_iris()
    iris_data = preprocessing.normalize(iris.data)
    targets = []
    data_list = []
    for i in range(len(iris_data)):
        if iris.target[i] == 0:
            targets.append([1, 0, 0])
        elif iris.target[i] == 1:
            targets.append([0, 1, 0])
        else:
            targets.append([0, 0, 1])
        data_list.append(list(iris_data[i]))
    data = SourceData()
    data.data, data.target = data_list, targets
    return data


def get_dataset():
    i = input("Which dataset should I load? [1] Iris, [2] Cars, [3] Breast Cancer, [4] Votes, [5] Loans, [6] Lenses, [7] Pima ")
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

    if i == '7' or i.lower == 'pima':
        return load_pima_csv()


def cross_validate(data, targets, classifier, num_folds=10):
    accuracy = []
    subset_size = len(data)/num_folds
    for i in range(num_folds):
        testing_data, testing_targets = data[i*subset_size:][:subset_size], targets[i*subset_size:][:subset_size]
        training_data = data[:i*subset_size] + data[(i + 1)*subset_size]
        training_targets = targets[:i*subset_size] + targets[(i + 1)*subset_size]
        classifier.train(data=training_data, targets=training_targets)

        result = classifier.predict(testing_data)

        # Count the number right
        num_right = 0
        for predicted, actual in zip(result, testing_targets):
            num_right += 1 if classifier.check_output(predicted, actual) else 0

        accuracy.append(num_right / len(testing_targets))

    overall_accuracy = 0
    for a in accuracy:
        overall_accuracy += a
    overall_accuracy /= len(accuracy)

    return overall_accuracy


def main():
    source = get_dataset()

    train_data, test_data, train_targets, test_targets = cross_validation.train_test_split(source.data, source.target,
                                                                                           test_size=1-get_split_percentage())

    train_bunch = Bunch()
    train_bunch['data'], train_bunch['target'] = train_data, train_targets
    train_permutations = []
    test_permutations = []

    for i in range(300):
        temp_train = Bunch()
        temp_test = Bunch()
        temp_train['data'], temp_test['data'], temp_train['target'], temp_test['target'] = cross_validation.train_test_split(train_bunch.data, train_bunch.target, test_size=0.3)
        train_permutations.append(temp_train)
        test_permutations.append(temp_test)

    # Test with our classifier
    tester = NeuralNet([4, len(train_permutations[0].target[0])],
                       num_inputs=int(len(train_permutations[0].data[0])),
                       learning_rate=get_learning_rate(NeuralNet.get_default_learning_rate()))

    for epoch in range(len(train_permutations)):
        tester.train(data=deepcopy(train_permutations[epoch].data), targets=deepcopy(train_permutations[epoch].target))

        result = tester.predict(test_permutations[epoch].data)

        # Count the number right
        num_right = 0
        for predicted, actual in zip(result, test_permutations[epoch].target):
            num_right += 1 if tester.check_output(predicted, actual) else 0

        accuracy = num_right * 100 / len(test_permutations[epoch].target)

        # Show Accuracy
        print("Epoch ", epoch, "Accuracy: %2.2f%%" % accuracy,)

    result = tester.predict(test_data)

    # Count the number right
    num_right = 0
    for predicted, actual in zip(result, test_targets):
        num_right += 1 if tester.check_output(predicted, actual) else 0

    accuracy = num_right * 100 / len(test_targets)

    # Show Accuracy
    print("Final Accuracy: %2.2f%%" % accuracy, )


if __name__ == "__main__":
    main()
