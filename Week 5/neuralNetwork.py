import numpy as np


class Perceptron(object):
    def __init__(self, num_inputs, output_function, error_function, speed=0.3):
        self.weights = np.random.uniform(-0.5, 0.5, num_inputs)
        self.learn_speed = speed
        self.output = output_function
        self.error = error_function

    def __str__(self, id_num):
        ret = "[N" + str(id_num) + "] = {"
        for i in range(len(self.weights)):
            ret += "w" + str(i) + ": " + str(self.weights[i]) + ", " if i is not 0 else " "
        ret += "}"
        return ret

    def adjust_weights(self, expected, output, forward_weights, inputs):
        """
        This function adjusts the weights on the inputs
        :param expected: the expected output. Either a 1 or a 0
        :param output: the perceptron's output. Either a 1 or a 0
        :param forward_weights: The weights of the last visited nodes
        :param inputs: the inputs for the node
        :type expected: int
        :type output: int
        :type inputs: list
        :return:
        """
        # assert len(self.weights) == len(inputs)
        weight_copy = self.weights[:]
        error = self.error(output, forward_weights, expected)
        for i in range(len(self.weights)):
            self.weights[i] -= self.learn_speed * inputs[i] * error
        return weight_copy, error

    def process(self, inputs):
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
        # assert len(inputs) == len(self.weights)
        for i in range(len(self.weights)):
            output += inputs[i] * self.weights[i]
        output = self.output(output)
        return output


class Layer(object):
    def __init__(self, num_neurons, bias, num_inputs, learning_rate, functions=0):
        """
        Initialize the layer and it's function in the network.
        :param num_neurons: Number of neurons in the layer
        :param bias: The bias for that layer
        :param num_inputs: Number of inputs each neuron will receive
        :param learning_rate: Float of how fast the neuron should learn
        :param functions: Either a 1 or a 0. 0 if it is the output layer, 1 if it is a hidden layer
        :type num_neurons: int
        :type bias: float
        :type num_inputs: int
        :type learning_rate: float
        :type functions int
        """
        self.neurons = []
        for i in range(num_neurons):
            if functions == 0:
                self.neurons.append(Perceptron(num_inputs + 1,
                                               speed=learning_rate,
                                               output_function=self.sig_output,
                                               error_function=self.end_error))
            else:
                self.neurons.append(Perceptron(num_inputs + 1,
                                               speed=learning_rate,
                                               output_function=self.sig_output,
                                               error_function=self.hidden_error))
        self.bias = float(bias)
        self.num_inputs = num_inputs
        self.functions = functions

    def __str__(self, index):
        ret = "[L" + str(index) + "] = [\n"
        for j, neuron in enumerate(self.neurons):
            ret += neuron.__str__(j) + "\n"
        ret += "=> " + "Output" if self.functions == 0 else "Hidden"
        ret += "]\n"
        return ret

    def get_output(self, inputs):
        output = []
        inputs += [self.bias]
        for index, neuron in enumerate(self.neurons):
            output.append(neuron.process(inputs))
        return output

    def update(self, outputs, errors, layer_forward_weights, inputs, is_hidden):
        layer_weights = []
        layer_errors = []
        for index, neuron in enumerate(self.neurons):
            neuron_forward_weights = []
            for i in range(len(layer_forward_weights)):
                neuron_forward_weights.append(layer_forward_weights[i][index])
            layer_weight, layer_error = neuron.adjust_weights(errors[index] if not is_hidden else errors,
                                                              outputs[index],
                                                              neuron_forward_weights,
                                                              inputs)
            layer_weights.append(layer_weight)
            layer_errors.append(layer_error)
        return layer_weights, layer_errors

    @staticmethod
    def sig_output(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def end_error(a, weights, target):
        return a * (1 - a) * (a - target)

    @staticmethod
    def hidden_error(a, weights, errors):
        sum_errors = 0
        assert len(weights) == len(errors)
        for index in range(len(weights)):
            sum_errors = weights[index] * errors[index]
        return a * (1 - a) * sum_errors


class NeuralNet(object):
    def __init__(self, layer_params=[3], bias=-1, num_inputs=3, learning_rate=0.1):
        """

        :param layer_params: An array of the number of Neurons in each layer. (e.g. [3, 5, 1]
        :param bias: Number to set for the bias. Default is -1
        :param num_inputs: Number of columns in the data that will be passed to the train function
        :type bias: float
        :type num_inputs: int
        """
        self.layers = []
        for i, layer_height in enumerate(layer_params):
            self.layers.append(Layer(layer_height,
                                     bias,
                                     layer_params[i - 1] if i > 0 else num_inputs,
                                     learning_rate,
                                     0 if len(layer_params) - 1 == i else 1))
        self.bias = float(bias)
        self.num_inputs = num_inputs

    def __str__(self):
        ret = "NeuralNetwork: \n"
        for i in range(len(self.layers)):
            ret += self.layers[i].__str__(i)
        return ret

    @staticmethod
    def get_default_learning_rate():
        return 0.1

    @staticmethod
    def check_output(output, target):
        if len(output) is 0:
            return False
        elif len(output) is not len(target):
            print("Output and target have different lengths!!!!", flush=True)
        else:
            largest = 0
            for i in range(len(output)):
                largest = output[i] if output[i] > largest else largest
            for i in range(len(output)):
                output[i] = 0 if output[i] != largest else 1
            for i in range(len(output)):
                if output[i] != target[i]:
                    return False
            return True

    def train(self, data, targets):
        for row in range(len(data)):
            layer_output = []
            for layer in range(len(self.layers)):
                layer_output.append([])
                layer_output[layer] = self.layers[layer].get_output(layer_output[layer - 1] if layer > 0 else data[row])
            # print("Data_train ", row, ": ", self)
            layer_forward_weights = []
            layer_forward_errors = []
            for layer in reversed(range(len(self.layers))):
                layer_forward_weights, layer_forward_errors = self.layers[layer].update(
                    errors=targets[row] if layer == len(self.layers) - 1 else layer_forward_errors,
                    outputs=layer_output[layer],
                    layer_forward_weights=layer_forward_weights,
                    inputs=layer_output[layer - 1] if layer != 0 else data[row],
                    is_hidden=True if layer != len(self.layers) - 1 else False)
            # print("Data_train_update ", row, ": ", self)

    def predict(self, data):
        output = []
        # print(self)
        for row in range(len(data)):
            layer_output = []
            for layer in range(len(self.layers)):
                layer_output.append([])
                layer_output[layer] = self.layers[layer].get_output(layer_output[(layer - 1)] if layer > 0 else data[row])
            # print("Data_predict ", row, ":", layer_output)
            output.append(layer_output[len(layer_output) - 1])
        return output
