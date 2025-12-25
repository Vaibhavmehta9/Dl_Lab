# XOR Gate using Shallow Neural Network (NumPy)

This project implements a simple shallow neural network to solve the XOR logic gate using only NumPy.
No machine learning or deep learning libraries are used. The purpose of this project is to understand
how neural networks work internally.

## About the Project
The XOR problem cannot be solved using a single neuron because it is not linearly separable.
To solve this, a neural network with one hidden layer is used.

## Network Details
- Input layer: 2 neurons  
- Hidden layer: 2 neurons (Sigmoid activation)  
- Output layer: 1 neuron (Sigmoid activation)

## Technologies Used
- Python  
- NumPy  

## Training Details
- Epochs: 1000  
- Learning rate: 0.1  

## Result
After training, the network correctly predicts the XOR output:
