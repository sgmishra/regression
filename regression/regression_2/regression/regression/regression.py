#!/usr/bin/env python
# coding: utf-8

# In[73]:


import requests
import pandas as pd
import scipy
import numpy as np
import sys
from scipy import stats


# In[11]:


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


# In[78]:



def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    train=pd.read_csv(TRAIN_DATA_URL, index_col=0, header=None).T
    x=np.array(list(train['area']))
    y=np.array(list(train['price']))
    slope, intercept, r_value, p_value, std_err=stats.linregress(x, y)
    area=np.array(area)
    area=area*slope
    area=area+intercept
    return area


# In[79]:


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = np.array(list(validation_data.keys()))
    prices = np.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = np.sqrt(np.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")

