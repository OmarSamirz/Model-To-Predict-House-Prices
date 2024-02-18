import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def rmse_cv(model, X, y):
    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(X.values)
    rmse= np.sqrt(-cross_val_score(model, X.values, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)