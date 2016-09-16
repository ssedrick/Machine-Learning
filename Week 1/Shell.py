import sys
import random
import numpy as np
from sklearn import datasets
from hardCoded import HardCoded


def check_accuracy(guess, real):
    if guess == real:
        return 1
    else:
        return 0


def get_split_percentage():
    percent = 0
    while percent < 100 and percent > 0:
        percent = input("What percentage of the dataset should be used for training? ")
        if percent < 100 and percent > 0:
            print("Sorry, I need a number between 100 and 0")

    return float(percent) / 100


def main(args):
    iris = datasets.load_iris()

    # Shuffle the data and target together
    shuffled = list(zip(iris.data, iris.target))

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
    tester = HardCoded()
    tester.train(training_data, training_target)

    result = tester.predict(test_data)

    # Count the number right
    num_right = 0
    for predicted, actual in zip(result, test_target):
        num_right += check_accuracy(predicted, actual)

    # Show Accuracy
    print("Accuracy: %2.2f%%" % (num_right / len(test_target)))

if __name__ == "__main__":
    main(sys.argv)
