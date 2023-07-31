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
            return np.transpose(self.activation(np.dot(self.weights, np.transpose(X))))
    
    'Calculates the cost of layer'
    def calculate_error(self, X, output, next_error = None):
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
            if next_error is None:
                return -1
            z = np.dot(self.weights, np.transpose(X))
            ones = np.ones((1, z.shape[2]))
            z = np.append(ones, z, 0)
            return np.multiply(np.dot(np.transpose(self.weights), next_error), z)
