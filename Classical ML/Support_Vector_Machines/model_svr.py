import numpy as np

class SupportVectorRegression:
    
    def __init__(self, learning_rate, iterations, epsilon, lambda_param):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.epsilon = epsilon
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None
    
    def _calc_gradients(self, X: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if abs(error) > self.epsilon:
            dw = 2 * self.lambda_param * self.weights + (np.sign(error) * X)
            db = np.sign(error)
        else:
            dw = 2 * self.lambda_param * self.weights
            db = 0
        return dw, db
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            for idx, X_idx in enumerate(X):
                prediction = X_idx @ self.weights + self.bias
                error = prediction - y[idx]
                dw, db = self._calc_gradients(X_idx, error)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def prediction(self, X: np.ndarray) -> float:
        return X @ self.weights + self.bias
