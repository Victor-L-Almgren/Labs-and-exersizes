#libraries
import numpy as np
from tqdm.notebook import tqdm_notebook

#Neural network
class Dense:
    def __init__(self, input_size, output_size, weight_initialization_range=(-0.5, 0.5), bias_initialization_range=(-0.5, 0.5)):
        self.weights = np.random.uniform(weight_initialization_range[0], weight_initialization_range[1], size=(output_size, input_size))   
        self.biases = np.random.uniform(bias_initialization_range[0], bias_initialization_range[1], size=(output_size, 1))    

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.biases 

    def backward(self, output_gradient, learning_rate):     
        weights_gradient = np.dot(output_gradient, self.input.T)
        biases_gradient = output_gradient
        input_gradient = np.dot(self.weights.T, output_gradient)
        
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient
        return input_gradient

class Sigmoid:
    def forward(self, input):
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, output_gradient, learning_rate):
        sig_derivative =  self.output * (1 - self.output)
        return output_gradient * sig_derivative

class Tanh:
    def forward(self, input):
        self.output = np.tanh(input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        tanh_derivative = (1 - self.output ** 2)
        return output_gradient * tanh_derivative

class Relu:
    def forward(self, input):
        self.output = np.maximum(0, input)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        relu_derivative =  (self.output > 0)
        return output_gradient * relu_derivative

class NeuralNetwork:
    def __init__(self, network):
        self.network = network
        self.error_history = []

    @staticmethod
    def mse(y_true, y_pred):
        error = np.mean(np.power(y_true - y_pred, 2))
        return error

    @staticmethod
    def mse_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)

    @staticmethod
    def exp_decay(epoch, input_learning_rate, exp_decay_k):
        output_learning_rate = input_learning_rate * np.exp(-epoch*exp_decay_k)
        return output_learning_rate

    def predict(self, input_data):
        output = input_data
        for layer in self.network:
            output = layer.forward(output)
        return output

    def fit(self, x_train, y_train, learning_rate=0.01, max_iterations=1000, LRS=False, exp_decay_k=0.01, exp_decay_t=10):
        self.learning_rate_history = []
        self.learning_rate = learning_rate

        for epoch in tqdm_notebook(range(max_iterations), desc='Training'):
            error = 0
            for x, y in zip(x_train, y_train):
                output = self.predict(x)
                error += self.mse(y, output)
                grad = self.mse_derivative(y, output)
                
                for layer in reversed(self.network):
                    grad = layer.backward(grad, learning_rate)
                    
            error /= np.size(x_train)
            self.error_history.append(error)
        
            if LRS and epoch % exp_decay_t == 0:
                self.learning_rate_history.append(learning_rate)
                learning_rate = self.exp_decay(epoch, learning_rate, exp_decay_k)