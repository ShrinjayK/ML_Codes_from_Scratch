"""
This is a general implementation of a Linear Regression (for both Simple & Multiple) with the gradient descent updation of the model params.
"""

# 1. Importing the libraries
import numpy as np

# 2. Create a class called Linear Regression

class LinearRegression:

    # Create a constructor that is going to initialise all the necessary params to the model everytime that this class is called.
    def __init__(self, learning_rate = 0.01, iterations = 10000):

      """ There are 4 things that we need to initialise before be carry forward with the implementation:
        1. The Learning Rate: Decides the rate at which at which the parameters would be udpated during the training loop
        2. Iterations       : Total number of times the model would be updating the values of the parameters to better fit the data.
        3. Weights          : The inidvidual weights of the independent features -> also called the slope of the line that we are fitting to the data.
        4. Bias             : The baseline value that the model needs to predict if there is no data available for the independent features.
        """
      self.learning_rate = learning_rate
      self.iterations = iterations
      self.weights = None
      self.bias = None
    
    # Now we implement the function that is going to perform the calculation of the gradients of the loss function (In this case MSE, for simplicity) with respect to the parameters of the model, namely - self.weights and self.bias
    def _calculate_gradients(self, X: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

      """" Gradients are the partial derivatives of the loss function (MSE) with respect to the parameters.
      1.  dw - The gradient of the loss function with respect to self.weights
      2.  db - The gradient of the loss function with respect to self.bias
      - n    : Total number of samples in the training data `X`
      """ 
      n = X.shape[0]
      dw = (1 / n) * (X.T @ error)
      db = (1 / n) * np.sum(error)
      return dw, db
    
    # No we can implement the training loop or the `fit` function for the model
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
      """
      - n_features  : Total number of independent features in the training dataset `X`
      - y_hat       : The predicted value of `y` on the basis of the values of the `ith` iteration of the self.weights and self.bias
      - X           : Independent features of the Training dataset (generally called `X_train`)
      - y           : Dependent/Target feature of the Training dataset (generally called `y_train`)
      """
      n_features = X.shape[1]
      y = y.reshape(-1)             # Ensure that the y is a single 1D array before moving forward. Can also use the better version of this by using y = y.ravel()
      self.weights = np.zeros(n_features)
      self.bias = 0
      for _ in range(self.iterations):
         y_hat = X @ self.weights + self.bias
         error = y_hat - y
         dw, db = self._calculate_gradients(X, error)
         self.weights -= self.learning_rate * dw
         self.bias -= self.learning_rate * db
    
    # The final predict function that is going to return us the output of the entire model.
    def predict(self, X: np.ndarray) -> np.ndarray:
       return X @ self.weights + self.bias
    
    # The function to evaluate the model peformance
    def rmse_loss(self, y_hat: np.ndarray, y_true: np.ndarray) -> float:
       """
       - y_hat      : The predicted value of the dependent feature using the linear regression model.
       - y_true     : The ground truth value of the dependent feature (generally depicted as `y_test`)
       """
       mse = np.mean((y_true - y_hat) ** 2)
       return np.sqrt(mse)