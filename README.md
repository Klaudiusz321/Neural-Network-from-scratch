# Neural Network Implementation from Scratch

## Overview
A comprehensive neural network implementation in Python using NumPy. This project implements various components of deep learning, including different layer types, activation functions, optimizers, and loss functions.

## Features

### Layer Types
- **Dense Layer (`Layer_Dense`)**: Fully connected layer with weights and biases
- **Dropout Layer (`Layer_Dropout`)**: Regularization layer to prevent overfitting
- **Input Layer (`Layer_Input`)**: Handles input data processing

### Activation Functions
- **ReLU (`Activation_ReLU`)**: Rectified Linear Unit activation
- **Softmax (`Activation_Softmax`)**: For classification output
- **Sigmoid (`Activation_Sigmoid`)**: For binary classification
- **Linear (`Activation_Linear`)**: For regression tasks

### Optimizers
- **SGD (`Optimizer_SGD`)**: Stochastic Gradient Descent with momentum
- **Adagrad (`Optimizer_Adagrad`)**: Adaptive gradient algorithm
- **RMSprop (`Optimizer_RMSprop`)**: Root Mean Square propagation
- **Adam (`Optimizer_Adam`)**: Adaptive Moment Estimation

### Loss Functions
- **Categorical Cross-Entropy (`Loss_CategoricalCrossentropy`)**: For multi-class classification
- **Binary Cross-Entropy (`Loss_BinaryCrossentropy`)**: For binary classification
- **Mean Squared Error (`Loss_MeanSquaredError`)**: For regression tasks
- **Mean Absolute Error (`Loss_MeanAbsoluteError`)**: Alternative regression loss

### Regularization
- L1 regularization (weights and biases)
- L2 regularization (weights and biases)
- Dropout regularization

### Accuracy Metrics
- **Categorical Accuracy**: For classification tasks
- **Regression Accuracy**: For regression tasks

## Dependencies
```python
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
```

## Usage Example

### 1. Create and Prepare Data
```python
X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)
```

### 2. Build Model
```python
model = Model()
model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512, 3))
model.add(Activation_Softmax())
```

### 3. Configure Model
```python
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
    accuracy=Accuracy_Categorical()
)
```

### 4. Train Model
```python
model.finalize()
model.train(X, y, 
    validation_data=(X_test, y_test),
    epochs=10000, 
    print_every=100
)
```

## Model Architecture Features

### Forward Pass
- Input processing
- Layer-by-layer forward propagation
- Activation function application
- Loss calculation

### Backward Pass
- Gradient computation
- Backpropagation through layers
- Parameter updates via optimizers

### Training Process
- Batch processing
- Loss calculation with regularization
- Accuracy monitoring
- Validation checking
- Learning rate decay

## Advanced Features

### Regularization Implementation
```python
# L1 regularization
if layer.weight_regularizer_l1 > 0:
    regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

# L2 regularization
if layer.weight_regularizer_l2 > 0:
    regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
```

### Dropout Implementation
```python
class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs, training):
        if not training:
            self.output = inputs.copy()
            return
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask
```

## Output Example
```
epoch: 100, acc: 0.750, loss: 0.683 (data_loss: 0.618, reg_loss: 0.065), lr: 0.0499
validation, acc: 0.793, loss: 0.654
```

## Best Practices
1. Use appropriate learning rates (typically 0.001 to 0.05)
2. Apply regularization for large networks
3. Monitor validation metrics to prevent overfitting
4. Use dropout in deeper networks
5. Implement learning rate decay for convergence

## Limitations
- CPU-only implementation
- Limited to feed-forward neural networks
- No convolutional or recurrent layers
- Basic optimization techniques

## Future Improvements
- Add convolutional layers
- Implement batch normalization
- Add early stopping
- Include model saving/loading
- Add more activation functions
- Implement mini-batch processing