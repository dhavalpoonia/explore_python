import numpy as np 
from sklearn.model_selection import train_test_split


class Linear_Regression():

    def __init__(self,learning_rate, n_iters):
        self.learning_rate = learning_rate 
        self.n_iters = n_iters
        

    def fit(self, X, Y):

        # number of records and number of features 
        self.n_records, self.n_features = X.shape # no of rows and columns

        # initiate weights and bias 
        self.weights = np.zeros(self.n_features) 
        self.bias = 0

        self.X = X 
        self.Y = Y 

        # implement gradient descent 
        for i in range(self.n_iters):
            self.update_weights()


    def update_weights(self):
        y_prediction = self.predict(self.X)

        #calculate gradients 
        dw = -(2*(self.X.T).dot(self.Y - y_prediction))/self.n_records
        db = -(2*(self.Y - y_prediction).sum())/self.n_records

        # update weights and bias 
        self.weights = self.weights - self.learning_rate * dw
        self.bias = self.bias - self.learning_rate * db

    def predict(self,X):
        return X.dot(self.weights) + self.bias
    

