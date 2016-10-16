import numpy as np


class Perceptron(object):
    def __init__(self, num_inputs, speed=0.3):
        self.weights = np.random.random(num_inputs)
        print(self.weights, "\n")
        self.learn_speed = speed

    def __str__(self, id_num):
        ret = "[N" + str(id_num) + "] = {"
        for i in range(len(self.weights)):
            ret += "w" + str(i) + ": " + str(self.weights[i]) + ", " if i is not 0 else " "
        ret += "}"
        return ret

    def adjust_weights(self, expected, output):
        """
        This function adjusts the weights on the inputs
        :param expected: the expected output. Either a 1 or a 0
        :param output: the perceptron's output. Either a 1 or a 0
        :type expected: int
        :type output: int
        :return:
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learn_speed * (output - expected)

    def process(self, inputs, learning=-1):
        """
        Process the inputs to see if the neuron fires or not. Alternatively, learn from the expected value if an
        expected value is provided.
        :param inputs: list of inputs. Inputs must be floats
        :param learning: the expected output if the perceptron is supposed to be learning. Either a 1 or a 0 if learning
        :type inputs: list
        :type learning: int
        :return:
        """
        output = 0
        for i in range(len(inputs)):
            output += inputs[i] * self.weights[i]
        output = 1 if output > 0 else 0
        if learning is not -1 and output != learning:
            self.adjust_weights(learning, output)
        return output


class NeuralNet(object):
    def __init__(self, layer_params=[3], bias=-1, num_inputs=3):
        """

        :param layer_params: An array of the number of Neurons in each layer. (e.g. [3, 5, 1]
        :param bias: Number to set for the bias. Default is -1
        :param num_inputs: Number of columns in the data that will be passed to the train function
        :type bias: float
        :type num_inputs: int
        """
        self.network = []
        for i, layer in enumerate(layer_params):
            self.network.append([])
            for j in range(layer):
                self.network[i].append(Perceptron(layer_params[(i - 1)] if i > 0 else num_inputs + 1, 0.3))
        self.bias = bias
        self.num_inputs = num_inputs

    def __str__(self):
        ret = "NeuralNetwork: \n"
        for i in range(len(self.network)):
            ret += "[L" + str(i) + "] = [\n"
            for j, neuron in enumerate(self.network[i]):
                ret += neuron.__str__(j) + "\n"
            ret += "]\n"
        return ret

    def check_output(self, output, target):
        if len(output) is 0:
            return False
        elif len(output) is not len(target):
            print("Output and target have different lengths!!!!", flush=True)
        else:
            for i in range(len(output)):
                if output[i] != target[i]:
                    return False
            return True

    def train(self, data, targets):
        for row in range(len(data)):
            layer_output = []
            for layer in range(len(self.network)):
                print("Layer ", layer, " output: ")
                layer_output.append([])
                while self.check_output(layer_output[layer], targets[row]) is False:
                    layer_output[layer] = []
                    for index, neuron in enumerate(self.network[layer]):
                        layer_output[layer].append(
                            neuron.process(data[row] if layer is 0 else layer_output[(layer - 1)],
                                           targets[row][index] if layer is 0 else -1))

            print(layer_output[len(layer_output) - 1], ": ", targets[row], "\n")

    def predict(self, data):
        output = []
        for row in range(len(data)):
            output.append([])
            for layer in range(len(self.network)):
                for index, neuron in enumerate(self.network[layer]):
                    output[row].append(
                        neuron.process(data[row] if layer is 0 else output[(layer - 1)]))
        return output
