import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#A1:Basic Function Modules
def summation_unit(inputs, weights):
    return np.dot(inputs, weights[1:]) + weights[0]  # weights[0] is bias

def activation_unit(y, activation_type='step'):
    if activation_type == 'step':
        return 1 if y > 0 else 0
    elif activation_type == 'bipolar_step':
        return 1 if y > 0 else (-1 if y < 0 else 0)
    elif activation_type == 'sigmoid':
        return 1 / (1 + np.exp(-y))
    elif activation_type == 'tanh':
        return np.tanh(y)
    elif activation_type == 'relu':
        return max(0, y)
    elif activation_type == 'leaky_relu':
        return max(0.01*y, y)
    else:
        raise ValueError("Invalid activation type")

def comparator_unit(actual, predicted):
    return actual - predicted
