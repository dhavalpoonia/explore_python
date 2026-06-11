

### implementation of Standard Scaler from scratch 



import numpy as np 
import pandas as pd


class StandardScaler():

    # standard scaler or z score normalization is calculated usind >> (Xi - mean)/standard deviation

    # define mean and std
    def __init__(self):
        self.mean_ = 0 
        self.std_ = 0 


    def fit(self, X):
        array_X = np.array(X)
        self.mean_ = np.nanmean(array_X, axis = 0) 
        self.std_ = np.nanstd(array_X, axis = 0)

        # if variance is zero then replace std deviation with 1 as default
        # because the feature has no information hence we are replacing it with zero > which also does contain any information
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)

        return self

    def transform(self, X):
        array_X = np.array(X) 

        scaled_X = (array_X - self.mean_)/self.std_ 
        return scaled_X
    
    def inverse_transform(self, scaled_X):
        array_X = np.array(scaled_X) 

        X = array_X*self.std_ + self.mean_
        return X


    
class MinMaxScaler():

    # normalization is calculated using >> (Xi - Xmin)/(Xmax - Xmin)

    # mean and std
    def __init__(self):
        self.min_ = 0 
        self.max_ = 0 


    def fit(self, X):
        array_X = np.array(X)
        self.min_ = np.nanmin(array_X, axis = 0)  
        self.max_ = np.nanmax(array_X, axis = 0) 

        # if values are constant then to keep denominator non zero we replace it with 1 
        # this way all scaled values will be zero > same as raw values with no information
        self.max_ = np.where(self.min_ == self.max_, self.max_ + 1.0, self.max_)

        return self

    def transform(self, X):
        array_X = np.array(X) 

        scaled_X = (array_X - self.min_)/(self.max_ - self.min_)

        return scaled_X
    
    def inverse_transform(self, scaled_X):
        array_X = np.array(scaled_X) 

        X = array_X*(self.max_ - self.min_) + self.min_
        return X


class RobustScaler():

    # normalization is calculated using >> (Xi - Xmin)/(Xmax - Xmin)

    # mean and std
    def __init__(self):
        
        self.median_ = 0 
        self.iqr_ = 0 


    def fit(self, X):

        array_X = np.array(X)
        self.median_ = np.nanmedian(array_X, axis = 0)  

        q_25 = np.quantile(array_X, 25, axis = 0)
        q_75 = np.quantile(array_X, 75, axis = 0)

        if q_25 == q_75:
            self.iqr_ = 1
        else:
            self.iqr_ = q_75 - q_25 

        return self

    def transform(self, X):

        array_X = np.array(X)
        scaled_X = (array_X - self.median_)/(self.iqr_)

        return scaled_X
    
    def inverse_transform(self, scaled_X):

        array_X = np.array(scaled_X) 
        X = array_X*(self.iqr_) + self.median_

        return X



