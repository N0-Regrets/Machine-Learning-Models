import pandas as pd
import numpy as np
from sklearn.preprocessing import add_dummy_feature

# to make this notebook's output stable across runs
np.random.seed(42)


class LinearRegression:
    
    
    def __init__(self):
        self.theta=None
    

    def fit( self ,X_train, y_train ,learning_rate =0.1 ,epochs=1000):

        # Convert DataFrame to NumPy array if needed
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
            y_train = y_train.reshape(-1, 1)


        # to make sure not to change the original data structures
        X = X_train.copy()
        y = y_train.copy()
        m = X_train.shape[0]  # number of instances
        n = X_train.shape[1]  # number of features
        

        # n+1 to add the bias parameter
        # initialize the weights with random number  
        self.theta= np.random.randn(n+1,1)

        X = add_dummy_feature(X_train)  # add x0 = 1 to each instance


        for epoch in range(epochs):
            
            # ( X @ self.theta ) is the predictions (m * n+1) @ (n+1 * 1) ==> (m * 1) 
            # ( ( X @ self.theta ) - y) is the difference between the model's prediction and and the labels (m * 1) - (m * 1)
            # 2 / m * X.T @ ( ( X @ self.theta ) - y) computes the gradient by multiplying
            # each element of the prediction error vector by its corresponding feature value and then by 2/m.
            # the line calculates the partial derivatives of the MSE function with respect to each feature in the model.
            gradients = 2 / m * X.T @ ( ( X @ self.theta ) - y)
            
            # update the parameters
            self.theta = self.theta - learning_rate * gradients


    def predict(self ,X):
        
        # to make sure not to change the original data structures
        X_copy = X.copy()
        X_copy = add_dummy_feature(X_copy)  # add x0 = 1 to each instance
        return X_copy @ self.theta 
