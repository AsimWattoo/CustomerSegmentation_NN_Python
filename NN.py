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
    def forward_propagation(self, X):
        if self.is_input:
            return X
        else:
            ones = np.ones((X.shape[0], 1))
            temp_x = np.append(ones, X, 1)
            return np.transpose(self.activation(np.dot(self.weights, np.transpose(temp_x))))
    
    #Calculates the cost of layer
    def calculate_error(self, X, output, next_error = None, next_weights = None) -> np.ndarray:
        m = X.shape[0] # (m, n)
        if self.is_output:
            # m -> Number of records
            # n -> Number of features
            # l -> Number of labels
            prediction = self.forward_propagation(X) # (m, l)
            error = np.zeros((self.num_neurons, m)) # (l, m)
            out = np.transpose(output) # (1, m)
            for i in range(self.num_neurons):
                temp_output = out == i # (1, m)
                temp_prediction = np.reshape(prediction[:, i], (1, -1)) # (1, m)
                error[i, :] = temp_prediction - temp_output
            return  error
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
    def calc_grad(self, prev_output, next_error, lamda: float):
        temp_weights = self.weights
        temp_weights[:, 1] = 0
        m = prev_output.shape[0]
        delta = np.transpose(np.dot(next_error, prev_output))
        delta = delta / m
        zeros = np.zeros((1, delta.shape[1]))
        delta = np.transpose(np.append(zeros, delta, 0))
        return delta + (lamda / m) * temp_weights

    # Calculates the back propagation
    def back_propagation(self, prev_output, next_error, lamda: float, alpha : float = 0.01):
        delta = self.calc_grad(prev_output, next_error, lamda)
        self.weights -= alpha * delta

    # Calculates the loss
    def loss(self, X: np.ndarray, y: np.ndarray, num_labels: int, lamda: float):
        m = y.shape[0]
        J = 0
        temp_theta = np.array(self.weights)
        temp_theta[:, 0] = 0
        prediction = np.transpose(self.forward_propagation(X))
        y = np.reshape(y, (-1, 1))
        ones = np.ones((m, 1))
        for i in range(0, num_labels):
            temp_y = y == i
            temp_prediction = np.reshape(prediction[i, :], (-1, 1))
            J += (1 / m) * np.sum(-np.multiply(temp_y, np.log(temp_prediction)) - np.multiply((1 - temp_y), np.log(ones - temp_prediction))) + (lamda / (2 * m)) * np.sum(np.square(temp_theta))
        return J

    #Checks whether the gradient value is correctly calculated or not
    def check_gradient(self, epsilon: float, prev_output: np.ndarray, y: np.ndarray, next_error: np.ndarray, num_labels: int, aplha: float, lamda: float):
        gradient = self.calc_grad(prev_output, next_error, lamda)
        numerical_grad = np.zeros(self.weights.shape)
        initial_weights = self.weights
        temp_weights = np.zeros(self.weights.shape)
        for r in range(0, temp_weights.shape[0]):
            for c in range(0, temp_weights.shape[1]):
                temp_weights[r, c] = epsilon
                self.weights = initial_weights + temp_weights
                cost1 = self.loss(prev_output, y, num_labels, lamda)
                self.weights = initial_weights - temp_weights
                cost2 = self.loss(prev_output, y, num_labels, lamda)
                numerical_grad[r, c] = (cost1 - cost2) / (2 * epsilon)
                temp_weights[r, c] = 0

        self.weights = initial_weights
        # return np.linalg.norm(gradient - numerical_grad) / np.linalg.norm(numerical_grad + gradient)
        return np.linalg.norm(np.abs(gradient - numerical_grad)) / np.linalg.norm(np.abs(gradient) + np.abs(numerical_grad))

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

def backward_propagate(layers: list[DenseLayer], X: np.ndarray, y: np.ndarray, alpha: float, lamda: float):
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
        layer.back_propagation(activations[i - 1], errors[i - 1], lamda, alpha)
            
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

        for j in range(layers.__len__() - 2, -1, -1):
            loss += (lamda / (2 * m)) * np.sum(np.square(layers[j].weights))

        epoch_message = f'Epoch: {i + 1}, Training Loss: {round(loss, 2)}'

        if validate:
            val_output = validation_X
            for j in range(layers.__len__() - 1):
                val_output = layers[j].forward_propagation(val_output)
            val_loss = layers[-1].loss(val_output, validation_y, num_labels, lamda)

            for j in range(layers.__len__() - 2, -1, -1):
                val_loss += (lamda / (2 * m)) * np.sum(np.square(layers[j].weights))

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

def check_gradient(layers: list[DenseLayer], epsilon: float, X: np.ndarray, y:np.ndarray, num_labels: int, alpha: float, lamda: float):
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

    return layers[-1].check_gradient(epsilon, activations[-2], y, errors[-1], num_labels, alpha, lamda)

