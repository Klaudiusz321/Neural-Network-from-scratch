# Neural Network Implementation

A Python implementation of a neural network from scratch using NumPy. This implementation includes various optimization algorithms and activation functions.

## Features

- Dense layer implementation with regularization support (L1 and L2)
- Multiple activation functions:
  - ReLU (Rectified Linear Unit)
  - Softmax
- Various optimizers:
  - SGD (Stochastic Gradient Descent) with momentum
  - Adagrad
  - RMSprop
  - Adam
- Loss functions:
  - Categorical Cross-entropy
- Support for regularization:
  - L1 regularization
  - L2 regularization

## Dependencies

- NumPy
- NNFS (Neural Networks from Scratch)

## Code Structure

### Main Components

1. **Layer_Dense**: Implementation of a fully connected neural network layer
   - Supports forward and backward propagation
   - Includes L1 and L2 regularization

2. **Activation Functions**:
   - `Activation_ReLU`: ReLU activation function
   - `Activation_Softmax`: Softmax activation for classification

3. **Optimizers**:
   - `Optimizer_SGD`: Standard SGD with momentum support
   - `Optimizer_Adagrad`: Adaptive Gradient algorithm
   - `Optimizer_RMSprop`: Root Mean Square Propagation
   - `Optimizer_Adam`: Adaptive Moment Estimation

4. **Loss Functions**:
   - `Loss`: Base class with regularization support
   - `Loss_CategoricalCrossentropy`: For classification tasks
   - `Activation_Softmax_Loss_CategoricalCrossentropy`: Combined Softmax activation and Cross-entropy loss

## Usage Example

```python
# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create model layers
dense1 = Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(512, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create optimizer
optimizer = Optimizer_Adam(learning_rate=0.02, decay=5e-7)

# Training loop
for epoch in range(10001):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    
    # Calculate loss
    loss = loss_activation.forward(dense2.output, y)
    
    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    # Update weights
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
```

## Model Performance

The model includes validation testing and prints:
- Training accuracy and loss every 100 epochs
- Final validation accuracy and loss
- Separate tracking for data loss and regularization loss

## Advanced Features

1. **Learning Rate Decay**: All optimizers support learning rate decay
2. **Momentum**: Available in SGD optimizer
3. **Regularization**: Both L1 and L2 regularization supported
4. **Adaptive Learning**: Implementation of modern adaptive optimizers

## Note

This implementation is designed for educational purposes to understand the inner workings of neural networks. For production use, consider using established frameworks like TensorFlow or PyTorch.