import numpy as np
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import nnfs

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_Relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0



class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
   
class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

# class Optimizer_SGD:
#     def __init__(self, learning_rate=1., decay=0., momentum=0.):
#         self.learning_rate = learning_rate
#         self.current_learning_rate = learning_rate
#         self.decay = decay
#         self.iterations = 0
#         self.momentum = momentum
#     def pre_update_params(self):
#         if self.decay:
#             self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
            
#     def update_parameters(self, layer):
#         if self.momentum:
#             if not hasattr(layer, 'weight_momentums'):
#                 layer.weight_momentums = np.zeros_like(layer.weights)
#                 layer.bias_momentums = np.zeros_like(layer.biases)
#             weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
#             layer.weight_momentums = weight_updates
#             bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
#             layer.bias_momentums = bias_updates
#         else:
#             weight_updates = -self.current_learning_rate * layer.dweights
#             bias_updates = -self.current_learning_rate * layer.dbiases
#         layer.weights += weight_updates
#         layer.biases += bias_updates
    
            
    
    # def post_update_params(self):
    #     self.iterations += 1

class Optimizer_Adam:
    def __init__(self, learning_rate=1., decay=0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_parameters(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.bias_momentums = np.zeros_like(layer.biases)
            
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
        
        
    def post_update_params(self):
        self.iterations += 1

X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 64)
activation1 = Activation_Relu()

dense2 = Layer_Dense(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7)


loss_history = []
accuracy_history = []
epochs_history = []
learning_rate = []

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    
    
    if not epoch % 100:
        print(f'epoch: {epoch}, '+
              f'acc: {accuracy:.3f}, '+
              f'loss: {loss:.3f}'+
              f' lr: {optimizer.current_learning_rate}')
        loss_history.append(loss)
        accuracy_history.append(accuracy)
        epochs_history.append(epoch)
        learning_rate.append(optimizer.current_learning_rate)

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_parameters(dense1)
    optimizer.update_parameters(dense2)
    optimizer.post_update_params()




X_test, y_test = spiral_data(samples = 100, classes= 3)
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test)

predictions = np.argmax(loss_activation.output, axis = 1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis =1)
accuracy = np.mean(predictions == y_test)

print(f'validation, acc: {accuracy: .3f}, loss: {loss: .3f}')

plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 2)
plt.plot(epochs_history, loss_history, 'b-', linewidth=2)
plt.title('Strata w czasie treningu')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.grid(True)

# Wykres dokładności (accuracy)
plt.subplot(1, 2, 2)
plt.plot(epochs_history, accuracy_history, 'r-', linewidth=2)
plt.title('Dokładność w czasie treningu')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_history, learning_rate, 'g-', linewidth=2)
plt.title('Learning rate w czasie treningu')
plt.xlabel('Epoka')
plt.ylabel('Learning rate')
plt.grid(True)

plt.tight_layout()

plt.show()  








        






















# class NeuralNetwork:
#     def __init__(self, input_size=784, hidden_layers=[512, 521], output_size=10):
#         self.input_size = input_size
#         self.hidden_layers = hidden_layers
#         self.output_size = output_size

#         self.weights = []
#         self.biases = []
#         # Input layer -> first hidden layer
#         self.weights.append(np.random.randn(input_size, hidden_layers[0]))
#         self.biases.append(np.random.randn(1, hidden_layers[0]))
#         # Hidden layers
#         for i in range(len(hidden_layers)-1):
#             self.weights.append(0.01 * np.random.randn(hidden_layers[i], hidden_layers[i+1]))
#             self.biases.append(np.zeros((1, hidden_layers[i+1])))
#         # Output layer
#         self.weights.append(0.01 * np.random.randn(hidden_layers[-1], output_size))
#         self.biases.append(np.zeros((1, output_size)))
#         # We'll use this cache to store intermediate activations during forward pass
#         self.cache = []

#     def forward(self, X):
#         self.cache = [X]  # store input
#         A = X
#         for i in range(self.num_layers):
#             Z = np.dot(A, self.weights[i]) + self.biases[i]
#             # Apply ReLU activation for hidden layers; last layer remains linear for MSE
#             if i < self.num_layers - 1:
#                 A = np.maximum(0, Z)
#             else:
#                 A = Z
#             self.cache.append(A)
#         return A

#     def backward(self, y, y_pred, learning_rate):
#         """
#         Performs backpropagation for a network trained with MSE loss.
#         Here we implement full backpropagation for all layers.
#         """
#         m = y.shape[0]  # number of examples
#         # For MSE, the derivative dL/dy_pred = 2 * (y_pred - y) / m
#         delta = 2 * (y_pred - y) / m  # shape: (m, output_size)

#         # Backpropagate from the output layer to the first hidden layer.
#         for i in reversed(range(self.num_layers)):
#             A_prev = self.cache[i]  # activation from the previous layer
#             Z = np.dot(A_prev, self.weights[i]) + self.biases[i]  # pre-activation for layer i
#             # For hidden layers, apply derivative of ReLU:
#             if i < self.num_layers - 1:
#                 dZ = delta * (Z > 0).astype(float)
#             else:
#                 dZ = delta  # no activation derivative on output layer
#             grad_W = np.dot(A_prev.T, dZ)
#             grad_b = np.sum(dZ, axis=0, keepdims=True)

#             # Update weights and biases
#             self.weights[i] -= learning_rate * grad_W
#             self.biases[i] -= learning_rate * grad_b

#             # Propagate delta to previous layer
#             delta = np.dot(dZ, self.weights[i].T)

#     def train(self, X, y, learning_rate=0.01, epochs=1000):
#         for epoch in range(epochs):
#             # Forward pass: compute predictions
#             y_pred = self.forward(X)
#             # Compute loss: Mean Squared Error (MSE)
#             loss = np.mean((y_pred - y) ** 2)
#             # Backward pass: compute gradients and update weights
#             self.backward(y, y_pred, learning_rate)
#             if epoch % 100 == 0:
#                 print(f"Epoch {epoch}, Loss: {loss:.4f}")

#     @staticmethod
#     def softmax(z):
#         exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
#         return exp_z / np.sum(exp_z, axis=1, keepdims=True)

#     @staticmethod
#     def cross_entropy_loss(probs, y):
#         N = y.shape[0]
#         loss = -np.sum(np.log(probs[np.arange(N), y] + 1e-12)) / N
#         return loss

#     @staticmethod
#     def cross_entropy_gradient(probs, y):
#         N = y.shape[0]
#         y_onehot = np.zeros_like(probs)
#         y_onehot[np.arange(N), y] = 1
#         grad = (probs - y_onehot) / N
#         return grad

# # Example usage code outside the class:
# if __name__ == "__main__":
#     # For training on MSE loss
#     np.random.seed(42)
#     X = np.random.rand(100, 4)  # 100 examples, 4 features
#     y = np.random.rand(100, 2)  # 100 examples, 2 outputs (regression with MSE)
    
#     nn = NeuralNetwork(input_size=4, hidden_layers=[5, 3], output_size=2)
#     nn.train(X, y, learning_rate=0.01, epochs=1000)
#     test_input = np.random.rand(1, 4)
#     test_output = nn.forward(test_input)
#     print("Test output (MSE model):", test_output)
    
#     # For classification with cross entropy loss,
#     # you would need your output layer to produce logits and
#     # then apply softmax externally, and adjust the backward pass accordingly.
#     # For example:
#     logits = np.array([[2.0, 1.0, 0.1, -1.2],
#                        [0.5, 2.2, -0.3, 0.0],
#                        [1.2, 0.1, 0.3, -0.5]])
#     y_true = np.array([0, 1, 2])
#     probs = NeuralNetwork.softmax(logits)
#     ce_loss = NeuralNetwork.cross_entropy_loss(probs, y_true)
#     ce_grad = NeuralNetwork.cross_entropy_gradient(probs, y_true)
#     print("Softmax probabilities:\n", probs)
#     print("Cross entropy loss:", ce_loss)
#     print("Gradient of cross entropy loss:\n", ce_grad)
