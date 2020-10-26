
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data = pd.read_csv('D:\\INFR3700U-FInal-Project\\train\\train.csv')

#encode the label to predict in this case the qaulity
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
quality_cat = data["Y"]
quality_cat_encoded = encoder.fit_transform(quality_cat)
data["Y"] = quality_cat_encoded
print(encoder.classes_)

#exploring the data



