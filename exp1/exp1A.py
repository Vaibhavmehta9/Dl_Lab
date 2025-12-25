import numpy as np


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])  


np.random.seed(1)

W1 = np.random.randn(2, 2)   

W2 = np.random.randn(2, 1)  
b2 = np.zeros((1, 1))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)


lr = 0.1
epochs = 10000

for _ in range(epochs):

    
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    y_pred = sigmoid(z2)

  
    error = y - y_pred


    d_output = error * sigmoid_derivative(y_pred)
    d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(a1)

  
    W2 += lr * np.dot(a1.T, d_output)
    b2 += lr * np.sum(d_output, axis=0, keepdims=True)

    W1 += lr * np.dot(X.T, d_hidden)
    b1 += lr * np.sum(d_hidden, axis=0, keepdims=True)


print("Final Output:")
print(np.round(y_pred))
