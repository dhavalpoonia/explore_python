import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from pathlib import Path

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
    



if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent

    #load data 
    data_path = BASE_DIR/ "datasets" / "salary_data.csv"
    df = pd.read_csv(data_path)
    
    print(f"Shape of data: {df.shape}")

    X = df.iloc[:,:-1].values
    Y = df.iloc[:,1].values

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

    # load model
    model = Linear_Regression(learning_rate=0.02, n_iters = 1000) 

    # fit data 
    model.fit(x_train, y_train)

    #get model parameters 
    print(f"model weights: {model.weights}")
    print(f"model bias: {model.bias}")

    y_test_pred = model.predict(x_test)
    
    plt.scatter(x_test, y_test, color = 'red')
    plt.plot(x_test, y_test_pred, color = 'blue')
    plt.xlabel('YearExperience')
    plt.ylabel('Salary')
    plt.title('Salary vs Experience')

    #save actual vs prediction 
    output_path = BASE_DIR/"output"
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path/"LR_actual_vs_prediction.png", dpi = 300, bbox_inches = 'tight')
