import numpy as np


biases=[3, 4, 5]
weights=[[0.3, 0.5, 0.1],
         [0.4, 0.2, 0.9],
         [0.1, 0.8, 0.7]]
inputs=[1, 2, 3]


layer_outputs = []

for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)
