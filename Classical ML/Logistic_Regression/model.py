import numpy as np

# A basic implementation of the a Binary Logistic Regression Model (0/1 class prediction)
class LogisticRegression:
    # The Constructor function for initializing the required parameters and hyperparameters.
    """
    self.learning_rate : Controls the strength of the update of the params
    self.iterations    : Total number of times the model would get updated.
    self.weights       : Weights / Co-efficients of the independent features.
    self.bias          : The default value of the linear model in the case that the values of the independent features is 0. (Also called the Intercept.)
    """
    def __init__(self, learning_rate = 0.01, iterations = 10000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    # The Sigmoid Function for transforming the values of the linear model into a range of [0, 1] which is going to classify the output as belonging to class 1 or 0.
    """
    z   : Linear Model that follows the formula y = Mx + C or y = W.X + C (where W = weight matrix and C = bias)
    """
    def _sigmoid(self, z):
        return (1 / 1 + np.exp(-z))
    
    # Function to calculate the gradients of the loss function with respect to the params of the model, namely - the weights and the biases of the model.
    """
    dw  : Partial derivative of the loss function (BCE) wrt the weight param
    db  : Partial derivative of the loss function (BCE) wrt the bias param
    """
    def calculate_gradients(self, X: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        dw = (1 / n) * (X.T @ error)
        db = (1 / n) * np.sum(error)
        return dw, db
    
    # The Training loop of the Logisitic Regression model to train and update the weights and biases.
    """
    the `ravel()` function ensure that the `y` is a 1D array
    probs : probability values we get after passing the output of the linear model through the sigmoid function
    """
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_features = X.shape[1]
        y = y.ravel()
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.iterations):
            model = X @ self.weights + self.bias
            probs = self._sigmoid(model)
            error = probs - y
            dw, db = self.calculate_gradients(X, error)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    # Calculate the probabilties of using the values of the independent features. These are ranging between (-inf, +inf)
    def calculate_probs(self, X: np.ndarray) -> np.ndarray:
        model = X @ self.weights + self.bias
        return self._sigmoid(model)
    
    # Make the final predictions based on the value returned by the sigmoid. Assign all values greater than a `threshold` value to class `1` else to class `0`
    def predict(self, X: np.ndarray) -> int:
        probs = self.calculate_probs(X)
        return np.where(probs >= 0.5, 1, 0)
    
    # Calculate the binary cross entropy loss using the `y` (dependent variable) feature values from the predictions and the testing datsets.
    """
    y_hat   : Predicted probabilities of the dependent feature
    y_true  : Ground truth probability associated with the dependent feature
    """
    def bce_loss(self, y_hat: np.ndarray, y_true: np.ndarray) -> float:
        return -((y_true) * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))