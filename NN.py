import numpy as np
from tqdm import tqdm

class DenseLayer():
    def __init__(self, num_neurons, inputs, activation, activation_prime, is_input = False, is_output = False):
        self.num_neurons = num_neurons
        self.inputs = inputs
        # TODO: Randomly initialize all the variables and set biases to 0
        self.weights = np.random.randn(num_neurons, inputs + 1) * 0.01
        self.activation = activation
        self.activation_prime = activation_prime
        self.is_output = is_output
        self.is_input = is_input

    #Feeds forward the neural network
    def forward_propagation(self, X, append: bool = True):
        if self.is_input:
            return X
        else:
            temp_x = X
            if append:
                ones = np.ones((X.shape[0], 1))
                temp_x = np.append(ones, X, 1)
            return np.transpose(self.activation(np.dot(self.weights, np.transpose(temp_x))))

    # Calculates the loss
    def loss(self, prediction: np.ndarray, y: np.ndarray, num_labels: int):
        m = prediction.shape[0]
        J = 0
        y = np.reshape(y, (-1, 1))
        for i in range(num_labels):
            temp_y = y == i
            temp_prediction = np.reshape(prediction[:, i], (-1, 1))
            J += (1 / m) * np.sum(-np.multiply(temp_y, np.log(temp_prediction)) - np.multiply((1-temp_y), np.log(1 - temp_prediction)))
        return J # + (lamda / (2 * m)) * np.sum(np.square(temp_theta))

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

# Calculates the gradient and updates the variables
def backward_propagate(layers: list[DenseLayer], X: np.ndarray, y: np.ndarray, alpha: float, lamda: float, num_labels: int, return_grad: bool = False):
    activations = []
    m = X.shape[0]
    output = X
    for layer in layers:
        output = layer.forward_propagation(output, False)
        if not layer.is_output:
            ones = np.ones((output.shape[0], 1))
            output = np.append(ones, output, 1)
        activations.append(output)

    errors = []
    deltas = []
    dz3 = np.zeros((m, num_labels))
    y = np.reshape(y, (-1, 1))
    for i in range(num_labels):
       temp_y = y == i
       temp_prediction = np.reshape(output[:, i], (-1, 1))
       dz3[:, i:i+1] = - (1 / m) * (temp_prediction - temp_y)

    errors.append(dz3)
    for i in range(layers.__len__() - 2, -1, -1):
        layer = layers[i]
        dZ_next = errors[0]
        temp_dZ_next = dZ_next
        if i < layers.__len__() - 2:
            temp_dZ_next = temp_dZ_next[:, 1:]

        # If the layer is not input layer then
        if not layer.is_input:
            temp_weights = layers[i + 1].weights
            dA = np.dot(temp_dZ_next, temp_weights)
            prev_A = np.transpose(activations[i - 1])
            z = np.transpose(layer.activation(np.dot(layer.weights, prev_A)))
            ones = np.ones((z.shape[0], 1))
            z = np.append(ones, z, 1)
            dZ = np.multiply(dA, layer.activation_prime(z))
            errors.insert(0, dZ)
        delta = np.transpose(np.dot(np.transpose(activations[i]), temp_dZ_next))
        deltas.insert(0, delta)
        if not return_grad:
            layers[i + 1].weights = layers[i + 1].weights + alpha * delta
    
    if return_grad:
        return deltas

#Checks whether the gradient value is correctly calculated or not
def check_gradient(layers: list[DenseLayer], epsilon: float, X: np.ndarray, y: np.ndarray, num_labels: int, alpha: float, lamda: float):
    gradient = backward_propagate(layers, X, y, alpha, lamda, num_labels, True)
    initial_weights = None
    total_grad = np.array([])
    first_run = True
    for layer in layers:
        if layer.is_input:
            continue
        if first_run:
            initial_weights = np.reshape(layer.weights, (-1, 1))
        else:
            initial_weights = np.append(initial_weights, np.reshape(layer.weights, (-1, 1)), 0)
        first_run = False
    
    for grad in gradient:
        total_grad = np.append(total_grad, np.reshape(grad, (-1,  1)))
    temp_weights = np.zeros(initial_weights.shape)
    numerical_grad = np.zeros(initial_weights.shape)
    for r in range(0, temp_weights.shape[0]):
        for c in range(0, temp_weights.shape[1]):
            temp_weights[r, c] = epsilon
            weights = initial_weights + temp_weights
            cost1 = calculate_loss(layers, weights, X, y, num_labels, lamda)
            weights = initial_weights - temp_weights
            cost2 = calculate_loss(layers, weights, X, y, num_labels, lamda)
            numerical_grad[r, c] = (cost1 - cost2) / (2 * epsilon)
            temp_weights[r, c] = 0

    set_weights(layers, initial_weights)
    # return np.linalg.norm(gradient - numerical_grad) / np.linalg.norm(numerical_grad + gradient)
    return np.linalg.norm(np.abs(total_grad - numerical_grad)) / np.linalg.norm(np.abs(total_grad) + np.abs(numerical_grad))

# Calculates loss
def calculate_loss(layers: list[DenseLayer], weights: np.ndarray, X: np.ndarray, y: np.ndarray, num_labels: int, lamda: float):
    # Setting new weights
    set_weights(layers, weights)
    # Forward Propagating the layers
    predictions = forward_propagate(layers, X)
    return layers[-1].loss(predictions, y, num_labels)

# Assigns the weights from the list
def set_weights(layers: list[DenseLayer], weights: np.ndarray):
    # Setting new weights
    weight_index = 0
    for layer in layers:
        if layer.is_input:
            continue
        weights_size = layer.weights.shape[0] * layer.weights.shape[1]
        layer.weights = np.reshape(weights[weight_index:weight_index + weights_size], layer.weights.shape)
        weight_index += weights_size

def train(layers: list[DenseLayer],
          X: np.ndarray,
          y: np.ndarray,
          num_labels: int,
          epochs: int,
          alpha: float,
          lamda: float,
          validation_X: np.ndarray,
          validation_y: np.ndarray,
          validate: bool = False,
          display_method= None,
          epoch_operation = None):
    history = {"loss": []}

    for epoch in range(epochs):
        prediction = forward_propagate(layers, X)
        loss = layers[-1].loss(prediction, y, num_labels)
        history["loss"].append(loss)

        # Doing back propagation
        backward_propagate(layers, X, y, alpha, lamda, num_labels, False)

        if display_method is None:
            print(f'Epoch: {epoch} -> Loss: {round(loss, 2)}')
        else:
            display_method(f'Epoch: {epoch} -> Loss: {round(loss, 2)}')

        if epoch_operation is not None:
            epoch_operation()

    return history

def softmax(z):
    return np.exp(z) / np.reshape(np.sum(np.exp(z), 1), (-1, 1))

def softmax_comps(z, comps):
    return np.exp(z) / np.sum(np.exp(comps))

def softmax_prime(z):
    # z = softmax(z)
    # Number of records
    m = z.shape[0]
    n = z.shape[1]
    prime = np.zeros((m, n))
    for i in range(m):
        comps = z[i, :]
        for j in range(n):
            if i == j:
                prime[i, j] = z[i, j], (1 - z[i, j])
            else:
                prime[i, j] = -z[i, j] * z[i, j]
    return prime
