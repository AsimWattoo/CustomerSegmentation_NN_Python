import numpy as np
from tqdm import tqdm

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
    def loss(self, prediction: np.ndarray, y: np.ndarray, num_labels: int, lamda: float):
        m = y.shape[0]
        J = 0
        temp_theta = np.array(self.weights)
        temp_theta[:, 0] = 0
        prediction = np.transpose(prediction)
        y = np.reshape(y, (-1, 1))
        ones = np.ones((m, 1))
        for i in range(0, num_labels):
            temp_y = y == i
            temp_prediction = np.reshape(prediction[i, :], (-1, 1))
            J += (1 / m) * np.sum(-np.multiply(temp_y, np.log(temp_prediction)) - np.multiply((1 - temp_y), np.log(ones - temp_prediction)))
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

#Calculates the cost of layer
def calculate_error(layer: DenseLayer, X, output, next_error = None, next_weights = None) -> np.ndarray:
    if layer.is_output:
        prediction = np.transpose(layer.forward_propagation(X, False)) # (1, l)
        error = np.zeros((layer.num_neurons, 1)) # (l, 1)
        out = output # (1, 1)
        for i in range(layer.num_neurons):
            temp_output = out == i # (1, 1)
            temp_prediction = prediction[i]
            error[i] = temp_prediction - temp_output
        return  error
    elif layer.is_input:
        return np.array([])
    else:
        if next_error is None or next_weights is None:
            return np.array([])
        z = np.dot(layer.weights, np.transpose(X))
        ones = np.zeros((1, z.shape[1]))
        z = np.append(ones, z, 0) # (n + 1, 1)
        error = np.multiply(np.dot(np.transpose(next_weights), next_error), layer.activation_prime(z))
        return error[1:, :]

# Calculates the gradient
def calc_grad(output, next_error):
    delta = np.dot(next_error, output)
    return delta

# Calculates the gradient and updates the variables
def backward_propagate(layers: list[DenseLayer], X: np.ndarray, y: np.ndarray, alpha: float, lamda: float, return_grad: bool = False):
    m = X.shape[0]
    deltas = []
    first_iter = True
    for i in range(m):
        activations = []
        output = np.array([X[i]])
        errors = []
        index = 1
        # Finding the outputs for each layer
        for layer in layers:
            output = layer.forward_propagation(output, False)
            if not layer.is_output:
                ones = np.ones((output.shape[0], 1))
                output = np.append(ones, output, 1)
            activations.append(output)
            index += 1

        error = None
        # Finding the errors
        for j in range(layers.__len__() - 1, -1, -1):
            layer = layers[j]
            if not layer.is_input: 
                error = calculate_error(layer, activations[j - 1], y[i], error, None if layer.is_output else layers[j + 1].weights)
                errors.insert(0, error)
        
        for j in range(layers.__len__() -2, -1, -1):
            layer = layers[j]
            if not layer.is_output:
                calc = calc_grad(activations[j], errors[j])
                if first_iter:
                    deltas.insert(0, calc)
                else:
                    deltas[j] += calc
        first_iter = False

    # Normalizing all the deltas
    for i in range(deltas.__len__()):
        deltas[i] = (1 / m) * deltas[i]
        if not return_grad:
            layers[i + 1].weights -= alpha * deltas[i]

    # Returning the gradients
    if return_grad:
        return deltas

#Checks whether the gradient value is correctly calculated or not
def check_gradient(layers: list[DenseLayer], epsilon: float, X: np.ndarray, y: np.ndarray, num_labels: int, alpha: float, lamda: float):
    gradient = backward_propagate(layers, X, y, alpha, lamda, True)
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
    return layers[-1].loss(predictions, y, num_labels, lamda)

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
    # with Live(save_dvc_exp=True) as live:
    # live.log_param("Learning Rate", learning_rate)
    history = {
        "loss": [],
    }
    m = X.shape[0]
    if validate:
        history['val_loss'] = []

    for i in tqdm(range(epochs)):

        if epoch_operation != None:
            epoch_operation()

        output = X
        # live.log_param('Epoch', i)
        for j in range(layers.__len__() - 1):
            output = layers[j].forward_propagation(output)
        loss = layers[-1].loss(output, y, num_labels, lamda)

        # for j in range(layers.__len__() - 2, -1, -1):
        #     loss += (lamda / (2 * m)) * np.sum(np.square(layers[j].weights))

        epoch_message = f'Epoch: {i + 1}, Training Loss: {round(loss, 2)}'

        if validate:
            val_output = validation_X
            for j in range(layers.__len__() - 1):
                val_output = layers[j].forward_propagation(val_output)
            val_loss = layers[-1].loss(val_output, validation_y, num_labels, lamda)

            # for j in range(layers.__len__() - 2, -1, -1):
                # val_loss += (lamda / (2 * m)) * np.sum(np.square(layers[j].weights))

            epoch_message += f", Validation Loss: {round(val_loss, 2)}"
            history['val_loss'].append(val_loss)

        if display_method is None:
            print(epoch_message)
        else:
            display_method(epoch_message)

        history['loss'].append(loss)
        # live.log_metric("Loss", loss)
        backward_propagate(layers, X, y, alpha, lamda)

        # live.next_step()
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
