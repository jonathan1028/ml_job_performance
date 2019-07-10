import datetime
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

class Print(BaseEstimator, TransformerMixin):    
    def __init__(self, message, return_shape=False, columns=False):
        self.message = message
        self.return_shape = return_shape
        self.columns = columns
    
    def transform(self, X):
        shape = ""
        if(self.return_shape):
            shape = X.shape
        if(self.columns):
            shape = X.shape[1]
        print(self.message, shape)
        
        return X

    def fit(self, X, y=None, **fit_params):
        return self


class MissingValueRatioFilter(BaseEstimator, TransformerMixin):
    def __init__(self, ratio_missing):
        self.ratio_missing = ratio_missing
        self.selected_columns = []
        
    def transform(self, X):
        X = X[self.selected_columns]
        return X

    def fit(self, X, y=None, **fit_params):        
        # drop column if % of data missing is greater than this percentage
        temp = X.copy(deep=True)
        ratio = self.ratio_missing #0.1
        threshold = int(X.shape[0] * ratio)

        self.selected_columns = list(temp.dropna(thresh=len(temp)-threshold,axis=1).columns)
        
        return self

    
class ChangeDType(BaseEstimator, TransformerMixin):     
    def transform(self, X):
        X = X.astype(str)
        
        return X

    def fit(self, X, y=None, **fit_params):
        return self
    
class ConvertToDataFrame(BaseEstimator, TransformerMixin):
        
    def transform(self, X):
        X = pd.DataFrame(X.toarray())
        print("    Converted Matrix to DataFrame")

        return X

    def fit(self, X, y=None, **fit_params):
        return self

class ForceToNumerical(BaseEstimator, TransformerMixin):
    def transform(self, X):
        X['v71'] = pd.to_numeric(X['v71'], errors='coerce')
        
        return X

    def fit(self, X, y=None, **fit_params):
        return self
    
class HighCorrelationFilter(BaseEstimator,TransformerMixin):
    def __init__(self, correlation_decimal):
        self.selected_columns = []
        self.correlation_decimal = correlation_decimal
        
    def transform(self,X):
        print("      Selected Features:",len(self.selected_columns))
        X = X.iloc[:,self.selected_columns]
        
        return X
    
    def fit(self, X, y=None, **fit_params):
        test = X.iloc[:,:]
        corr = test.corr()
    
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[0]):
                if corr.iloc[i,j] >= self.correlation_decimal:
                    if columns[j]:
                        columns[j] = False

        self.selected_columns = list(test.columns[columns])
        
        return self

class StartTimer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.start_time = datetime.datetime.now()
    
    def transform(self, X):
        return X

    def fit(self, X, y=None, **fit_params):
        return self
    
    
class OutputRunTime(BaseEstimator, TransformerMixin):
    def __init__(self, start_time):
        self.start_time = start_time
        
    def transform(self, X):
        return X

    def fit(self, X, y=None, **fit_params):
        end_time = datetime.datetime.now()
        print(">>>Fit Time(seconds):", (end_time - self.start_time).total_seconds())
        return self