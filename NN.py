import numpy as np

class DenseLayer():
    def __init__(self, num_neurons, inputs, activation, activation_prime, is_input = False, is_output = False):
        self.num_neurons = num_neurons
        self.inputs = inputs
        self.weights = np.random.randn(num_neurons, inputs + 1)
        self.activation = activation
        self.activation_prime = activation_prime
        self.is_output = is_output
        self.is_input = is_input
    
    'Feeds forward the neural network'
    def forward_propagation(self, X):
        if self.is_input:
            return X
        else:
            ones = np.ones((X.shape[0], 1))
            temp_x = np.append(ones, X, 1)
            return np.transpose(self.activation(np.dot(self.weights, np.transpose(temp_x))))
    
    'Calculates the cost of layer'
    def calculate_error(self, X, output, next_error = None, next_weights = None):
        if self.is_output:
            m = X.shape[0]
            prediction = self.forward_propagation(X)
            error = np.zeros((self.num_neurons, m))
            out = np.transpose(output)
            for i in range(self.num_neurons):
                temp_output = out == i
                temp_prediction = np.reshape(prediction[:, i], (1, -1))
                error[i, :] = temp_prediction - temp_output
            return error
        else:
            if next_error is None or next_weights is None:
                return -1
            ones = np.ones((X.shape[0], 1))
            temp_x = np.append(ones, X, 1)
            z = np.dot(self.weights, np.transpose(temp_x))
            ones = np.ones((1, z.shape[1]))
            z = np.append(ones, z, 0)
            error = np.multiply(np.dot(np.transpose(next_weights), next_error), self.activation_prime(z))
            return error[1:, :]

    def back_propagation(self, prev_output, next_error):
        m = prev_output.shape[0]
        delta = np.transpose(np.dot(next_error, prev_output))
        delta = delta / m
        ones = np.ones((1, delta.shape[1]))
        delta = np.transpose(np.append(ones, delta, 0))
        self.weights -= delta


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def dummy_activation(z):
    return z


def forward_propagate(layers: list[DenseLayer], X):
    output = X
    for layer in layers:
        output = layer.forward_propagation(output)
    return output