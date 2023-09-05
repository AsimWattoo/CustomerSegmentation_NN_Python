import numpy as np
from tqdm import tqdm
import h5py

class DenseLayer():
    def __init__(self, num_neurons, inputs, activation, is_input = False, is_output = False):
        self.num_neurons = num_neurons
        self.inputs = inputs
        self.weights = np.random.randn(num_neurons, inputs)
        self.bias = np.zeros((num_neurons, 1))
        self.activation_name = activation
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'softmax':
            self.activation = softmax
            self.activation_prime = softmax_prime
        else:
            self.activation = dummy_activation
            self.activation_prime = dummy_activation
        self.is_output = is_output
        self.is_input = is_input

    #Feeds forward the neural network
    def forward_propagation(self, X):
        if self.is_input:
            return X, X
        else:
            Z = np.transpose(np.dot(self.weights, np.transpose(X)) + self.bias)
            A = self.activation(Z)
            return A, Z

    # Calculates the loss
    def loss(self, prediction: np.ndarray, y: np.ndarray, num_labels: int):
        m = prediction.shape[0]
        J = 0
        y = np.reshape(y, (-1, 1))
        for i in range(num_labels):
            temp_y = y == i
            temp_prediction = np.reshape(prediction[:, i], (-1, 1))
            J += - (1 / num_labels) * np.sum(temp_y * np.log(temp_prediction) + (1-temp_y) * np.log(1 - temp_prediction))
        return (1 / m) * J # + (lamda / (2 * m)) * np.sum(np.square(temp_theta))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def dummy_activation(z):
    return z

def forward_propagate(layers: list[DenseLayer], X):
    caches = []
    A = X
    for layer in layers:
        A, Z = layer.forward_propagation(A)
        caches.append((A, Z))
    return A, caches

# Calculates the gradient and updates the variables
def backward_propagate(layers: list[DenseLayer], y: np.ndarray, alpha: float, lamda: float, num_labels: int, caches, return_grad: bool = False):
    m = y.shape[0]
    # print(m)
    dZ_next = np.zeros((m, num_labels))
    y = np.reshape(y, (-1, 1))
    prediction = caches[-1][0]
    for i in range(num_labels):
        temp_y = y == i
        temp_prediction = np.reshape(prediction[:, i], (-1, 1))
        dZ_next[:, i:i+1] = (temp_prediction - temp_y)
    
    for i in range(layers.__len__() - 2, -1, -1):
        # print(f"Layer: {i + 1}")
        # print(f"dZ{i + 1} = {dZ_next}")
        # print(f"W{i+1} = {layers[i + 1].weights}")
        A, Z = caches[i]
        layer = layers[i]
        dW = (1 / m) * np.dot(A.T, dZ_next).T
        db = (1 / m) * np.reshape(np.sum(dZ_next, 0).T, (-1, 1))

        # print(f"dW{i + 1} = {dW}\n db{i+1} = {db}")
        if not layer.is_input:
            dA = np.dot(dZ_next, layers[i + 1].weights)
            # print(f"dA{i} = {dA}")
            # print(f"Z{i} = {Z}")
            # print(f'Z`{i} = {layers[i + 1].activation_prime(Z)}')
            dZ = np.multiply(dA, layers[i + 1].activation_prime(Z))
            dZ_next = dZ
        
        # Updating Weights
        layers[i + 1].weights -= alpha * dW
        layers[i + 1].bias -= alpha * db

#Checks whether the gradient value is correctly calculated or not
def check_gradient(layers: list[DenseLayer], epsilon: float, X: np.ndarray, y: np.ndarray, num_labels: int, alpha: float, lamda: float):
    gradient = backward_propagate(layers, y, alpha, lamda, num_labels,[], True)
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
          batch_size: int,
          alpha: float,
          lamda: float,
          validation_X: np.ndarray,
          validation_y: np.ndarray,
          validate: bool = False,
          display_method= None,
          epoch_operation = None):
    history = {"loss": [], "val_loss": []}
    m = X.shape[0]
    for epoch in tqdm(range(epochs)):
        startPoint = 0
        endPoint = batch_size

        batch_losses = []
        val_batch_losses = []

        while endPoint <= m:
            batch_X = X[startPoint:endPoint, :]
            batch_y = y[startPoint:endPoint]
            startPoint = endPoint
            endPoint = startPoint + batch_size

            # Forward Propagation
            prediction, caches = forward_propagate(layers, batch_X)

            # Calculating loss
            loss = layers[-1].loss(prediction, batch_y, num_labels)
            batch_losses.append(loss)

            # Doing back propagation
            backward_propagate(layers, batch_y, alpha, lamda, num_labels, caches, False)
        
        epoch_loss = np.average(batch_losses)
        history["loss"].append(epoch_loss)

        print_str = f'Epoch: {epoch} -> Loss: {round(epoch_loss, 2)}'

        # Validation
        if validate:
            startPoint = 0
            endPoint = batch_size
            val_m = validation_X.shape[0]
            while endPoint < val_m:
                batch_X = validation_X[startPoint:endPoint, :]
                batch_y = validation_y[startPoint:endPoint]
                startPoint = endPoint
                endPoint = startPoint + batch_size

                # Forward Propagation
                prediction, caches = forward_propagate(layers, batch_X)

                # Calculating loss
                loss = layers[-1].loss(prediction, batch_y, num_labels)
                val_batch_losses.append(loss)
            
            val_loss = np.average(val_batch_losses)
            history["val_loss"].append(val_loss)
            print_str += f', Validation Loss: {round(val_loss, 2)}'

        if epoch_operation is not None:
            epoch_operation()

        if display_method is None:
            print(print_str)
        else:
            display_method(print_str)

    return history

def softmax(z):
    shiftx = z - np.max(z)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

def softmax_prime(z):
    labels = z.shape[1]
    der = np.zeros(z.shape)
    for i in range(labels):
        I = np.eye(z.shape[0])
        x = np.reshape(z[:, i], (-1, 1))
        der[:, i:i+1] = np.dot((I - softmax(x).T), softmax(x))
    return der

# Saves the model to an h5 file
def save_model(layers: list[DenseLayer], filename: str):
    with h5py.File(filename, 'w') as file:
        file.create_dataset("layers", data= layers.__len__())
        for i in range(layers.__len__()):
            layer = layers[i]
            file.create_dataset(f"layer{i + 1}_weights", data= layer.weights)
            file.create_dataset(f"layer{i + 1}_bias", data= layer.bias)
            file.create_dataset(f"layer{i + 1}_activation", data=[layer.activation_name])
            file.create_dataset(f"layer{i + 1}_connection", data=[layer.is_input, layer.is_output])



# Loads the data from the model
def load_model(filename):
    layers = []
    with h5py.File(filename, 'r') as file:
        n = file['layers'][()]
        # Initializing all the layers
        for i in range(n):
            weights = file[f'layer{i + 1}_weights'][()]
            bias = file[f'layer{i + 1}_bias'][()]
            activation = file[f'layer{i + 1}_activation'][()].astype('U')[0]
            connections = file[f'layer{i + 1}_connection'][()]
            layer = DenseLayer(weights.shape[0], weights.shape[1], activation, connections[0], connections[1])
            layer.weights = weights
            layer.bias = bias
            layers.append(layer)
    return layers
    