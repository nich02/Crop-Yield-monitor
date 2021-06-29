from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("crop_data.csv")
df['pressure'].fillna(df['pressure'].median(), inplace=True)
df['yields'].fillna(df['yields'].median(), inplace=True)
le = LabelEncoder()
df['crop'] = le.fit_transform(df['crop'])
df['state'] = le.fit_transform(df['state'])

# Dealing with outliears
df['yields'] = df['yields'] < 1
features = df.drop(columns=['yields'])
target = df['yields']
sc = MinMaxScaler()
x_sc = sc.fit_transform(features)

lreg = LinearRegression()
# train test split
x_train, x_test, y_train, y_test = train_test_split(
    x_sc, target, test_size=0.3, random_state=0)
lreg.fit(x_train, y_train)
lreg_yhat = lreg.predict(x_test)
# Linear Regression
mse1 = np.mean((lreg_yhat - y_test)**2)
rmse1 = np.sqrt(mse1)
print(rmse1)
# Save the model
filename = 'cropmodel.pkl'
pickle.dump(lreg, open(filename, 'wb'))
