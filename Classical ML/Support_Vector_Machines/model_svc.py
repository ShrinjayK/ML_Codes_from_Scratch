import numpy as np

# A simple implementation of the SVC model that is going to output a 0/1 classification
class SupportVectorClassification:
    
    # The Constructor function that initialises the necessary params and hyperparams for the model.
    def __init__(self, learning_rate, iterations, lambda_param):
        """
        -   Learning_rate : Rate of learning per training loop. Controls the strength of the updates to the params.
        -   Iterations    : Total number of times that the model is going to update the params
        -   Lambda_param  : Regularization parameter
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None

    # The Training function
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        # Need to transform `y` since SVM requires -1 and 1 as inputs for the classification.
        y_transformed = np.where(y <= 0, -1, 1)
        for _ in range(self.iterations):
            for idx, X_idx in enumerate(X):

            # The condition for Hinge Loss: y_i * (w*X_i + bias) >= 1     <---- Very important ---->
                condition = y_transformed[idx] * (X_idx @ self.weights + self.bias) >= 1

                # If the condition holds then only apply the regularisation gradient
                if condition:
                    dw = 2 * self.lambda_param * self.weights
                    db = 0

                # If the condition doesn't hold then apply Regularisation gradient + the misclassification gradient
                else:
                    dw = 2 * self.lambda_param * self.weights - (X_idx @ y_transformed[idx])
                    db = -y_transformed[idx]

                # Update the weights and biases of the model using the gradients as calculated above.
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
    
    # The final prediction function that uses the final updated weights and bias and then passes the results to a `sign` function that return -1 / 1 for -ve / +ve respectively.
    def predict(self, X):
        approx = X @ self.weights + self.bias
        return np.sign(approx)
    

