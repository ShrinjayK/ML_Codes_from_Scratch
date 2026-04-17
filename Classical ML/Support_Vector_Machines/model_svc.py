import numpy as np

class SupportVectorClassifier:
    def __init__(self, learning_rate, iterations, lambda_param):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None
    
    def _calc_gradients(self, idx: int, X: np.ndarray, y: np.ndarray, condition: bool) -> tuple[np.ndarray, np.ndarray]:

        if condition:
            dw = 2 * self.lambda_param * self.weights
            db = 0
        else:
            dw = 2 * self.lambda_param * self.weights - (X @ y[idx])
            db = -y[idx]
        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_transformed = np.where(y <= 0, -1, 1)

        for _ in range(self.iterations):
            for idx, X_idx in enumerate(X):

                condition = (y_transformed[idx] - (X_idx @ self.weights + self.bias)) >= 1
                dw, db = self._calc_gradients(idx, X_idx, y_transformed, condition)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X: np.ndarray) -> int:
        approx = X @ self.weights + self.bias
        return np.sign(approx)
