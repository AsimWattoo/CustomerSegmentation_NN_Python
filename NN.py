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
    
    #Feeds forward the neural network
    def forward_propagation(self, X):
        if self.is_input:
            return X
        else:
            ones = np.ones((X.shape[0], 1))
            temp_x = np.append(ones, X, 1)
            return np.transpose(self.activation(np.dot(self.weights, np.transpose(temp_x))))
    
    #Calculates the cost of layer
    def calculate_error(self, X, output, next_error = None, next_weights = None) -> np.ndarray:
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
                return np.array([])
            ones = np.ones((X.shape[0], 1))
            temp_x = np.append(ones, X, 1)
            z = np.dot(self.weights, np.transpose(temp_x))
            ones = np.ones((1, z.shape[1]))
            z = np.append(ones, z, 0)
            error = np.multiply(np.dot(np.transpose(next_weights), next_error), self.activation_prime(z))
            return error[1:, :]

    # Calculates the gradient
    def calc_grad(self, alpha, prev_output, next_error):
        m = prev_output.shape[0]
        delta = np.transpose(np.dot(next_error, prev_output))
        delta = delta / m
        zeros = np.zeros((1, delta.shape[1]))
        delta = np.transpose(np.append(zeros, delta, 0))
        return delta

    # Calculates the back propagation
    def back_propagation(self, prev_output, next_error, alpha = 0.01):
        delta = self.calc_grad(alpha, prev_output, next_error)
        self.weights -= delta

    # Calculates the loss
    def loss(self, X: np.ndarray, y: np.ndarray, num_labels: int):
        J = 0
        m = X.shape[0]
        prediction = np.transpose(self.forward_propagation(X))
        for i in range(num_labels):
            temp_y = y == i
            temp_prediction = np.reshape(prediction[i, :], (-1, 1))
            J += (1 / m) * np.sum(-np.multiply(temp_y, np.log(temp_prediction)) - np.multiply((1 - temp_y), np.log(1 - temp_prediction)))
        return J

    #Checks whether the gradient value is correctly calculated or not
    def check_gradient(self, epsilon: float, prev_output: np.ndarray, y: np.ndarray, next_error: np.ndarray, num_labels: int, aplha: float):
        gradient = self.calc_grad(num_labels, prev_output, next_error)
        numerical_grad = np.zeros(self.weights.shape)
        initial_weights = self.weights
        temp_weights = np.zeros(self.weights.shape)
        for r in range(0, temp_weights.shape[0]):
            for c in range(0, temp_weights.shape[1]):
                temp_weights[r, c] = epsilon
                self.weights = initial_weights + temp_weights
                cost1 = self.loss(prev_output, y, num_labels)
                self.weights = initial_weights - temp_weights
                cost2 = self.loss(prev_output, y, num_labels)
                numerical_grad[r, c] = (cost1 - cost2) / (2 * epsilon)
                temp_weights[r, c] = 0

        self.weights = initial_weights
        return np.linalg.norm(numerical_grad + gradient) / np.linalg.norm(numerical_grad - gradient)

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

def backward_propagate(layers: list[DenseLayer], X, y, alpha):
    output = X
    activations = []
    for layer in layers:
        output = layer.forward_propagation(output)
        activations.append(output)

    errors = []
    next_error = []
    for i in range(layers.__len__() - 1, 0, -1):
        layer = layers[i]
        next_error = layer.calculate_error(activations[i - 1], y, None if layer.is_output else next_error, None if layer.is_output else layers[i + 1].weights)
        errors.insert(0, next_error)

    for i in range(layers.__len__() - 1, 0, -1):
        layer = layers[i]
        layer.back_propagation(activations[i - 1], errors[i - 1], alpha)
            