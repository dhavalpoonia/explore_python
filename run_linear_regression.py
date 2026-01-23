
import pandas as pd 
import matplotlib.pyplot as plt 
from pathlib import Path
from linear_regression import Linear_Regression


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
