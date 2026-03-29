# Multilayer Perceptron (MLP)

This project presents a **from-scratch implementation of a Multilayer Perceptron (MLP)** for both classification and regression tasks. Unlike typical frameworks (e.g., PyTorch, TensorFlow), the model is implemented entirely using NumPy, providing full control over:

- forward- and back-propagation,
- gradient descent optimization,
- activation and loss functions.

Additionally, the project includes **a Plotly-based browser interface** to visualize the evolution of weights and biases during training.

## Model

The neural network is implemented in a modular way with:

- fully connected, configurable layers,
- separate activation functions for hidden and output layers,
- configurable gradient-based optimization (batch learning, descent length),
- loss error functions of choice (e.g., MSE).

Each layer maintains its weight matrix, bias vector and intermediate activations (for backpropagation).

## Learning Algorithm

Details on implementation can be found in `MlpBase.py`.

### Forward Pass

The network utilises a standard, linear transformation (`z = W·x + b`), as well as nonlinear activation functions (e.g., sigmoid, ReLU).

### Backpropagation

Gradients are computed manually using the chain rule: output layer gradient is based on loss function, whereas hidden layers are updated via propagated error signals.

The implementation uses explicit formulas for layer-wise deltas and gradients.

### Optimization

Weights and biases are updated using gradient descent with configurable step size in the iterative process of learning over training samples.

### Supported Functions

The project includes multiple activation and loss functions (in `functions.py`):

Activation Functions:
- sigmoid,
- ReLU,
- linear,
- arctan.

Loss Functions:
- Mean Squared Error (MSE),
- Mean Absolute Error (MAE),
- cross-entropy with softmax.

## MNIST Dataset

The implementation is applied to the **MNIST dataset**, including:

- raw binary data parsing,
- normalization and one-hot encoding,
- training with stochastic mini-batches,
- accuracy tracking and confusion matrix visualization.

The MLP network achieved ~97% accuracy under conditions presented in `MnistAnalyzer.py`.


## Usage

To run training please utilize the code provided in `main.py`. Provide network parameters at the beginning of the file.

### Classification

```python
if __name__ == '__main__':
    train_in, train_out 
        = read_classification('data_classification/data.circles.train.500.csv')
    test_in, test_out 
        = read_classification('data_classification/data.circles.test.500.csv')

    model = peek(train_in, train_out, test_in, test_out)
    plot_classification(train_in, train_out, test_in, test_out, model)
```

### Regression

```python
if __name__ == '__main__':
    train_in, train_out 
        = read_regression('data_regression/data.activation.train.100.csv')
    test_in, test_out 
        = read_regression('data_regression/data.activation.test.100.csv')
    train_in, train_out, test_in, test_out 
        = normalize_regression(train_in, train_out, test_in, test_out)

    model = peek(train_in, train_out, test_in, test_out)
    plot_regression(train_in, train_out, test_in, test_out, model)
```