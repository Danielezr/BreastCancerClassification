import numpy as np
class NeuralNetwork:
    def __init__(self, n_features):
        np.random.seed(42)

        self.W = np.random.rand(n_features, 1) * 0.01 
        self.b = 0.0 #sesgo de bias

    def sigmoid(self, z):
        return np.where (
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )
    
    def forward(self, X):
        z = np.dot(X, self.W) + self.b # combinacion lineal
        
        y_hat = self.sigmoid(z) # activacion
        
        return y_hat