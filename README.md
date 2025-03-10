#Neural Network from Scratch - Spiral Data Classification
🧠 Project Overview
This project implements a simple neural network from scratch using only NumPy. The neural network is trained to classify points in the spiral dataset generated using the nnfs library. The model uses a fully connected architecture with ReLU and Softmax activation functions and is trained with Stochastic Gradient Descent (SGD).

🚀 Features
Custom Dense Layer: Implemented from scratch with forward and backward propagation.
Activation Functions: ReLU and Softmax with backpropagation.
Loss Function: Categorical Cross-Entropy.
Optimizer: Stochastic Gradient Descent (SGD) with learning rate control.
Training Visualization: Epoch-wise loss and accuracy tracking.
📊 Training Example
During training, the model outputs the accuracy and loss at regular intervals. Example output:

yaml
Copy
Edit
epoch: 0, acc: 0.333, loss: 1.098
epoch: 100, acc: 0.470, loss: 1.002
epoch: 1000, acc: 0.850, loss: 0.450
🔧 Installation
Clone the repository:
bash
Copy
Edit
git clone https://github.com/Klaudiusz321/Spiral-Neural-Network.git
Install dependencies:
bash
Copy
Edit
pip install ![training_history](https://github.com/user-attachments/assets/1342713f-e3ba-4f1c-a9e1-50660c789638)
numpy matplotlib nnfs
Run the script:
bash
Copy
Edit
python main.py
📈 Results
The network is capable of classifying the spiral dataset with high accuracy. The training process also generates plots showing loss and accuracy trends over epochs.

📂 Project Structure
bash
Copy
Edit
.
├── main.py                 # Main training script
├── requirements.txt        # Required Python packages
└── README.md               # Project documentation
💡 Future Improvements
Implementing additional optimizers (e.g., Adam, RMSprop)
Adding support for deeper neural networks
Visualizing decision boundaries in real-time during training
