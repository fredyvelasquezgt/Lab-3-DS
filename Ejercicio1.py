# Import required libraries
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score

df = pd.read_csv('train.csv')

X = df.iloc[:, 1:785]

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=40)
print(X_train.shape)
print(X_test.shape)

mlp = MLPClassifier(hidden_layer_sizes=(8, 8, 8),
                    activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train, y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

print(confusion_matrix(y_train, predict_train))
print(classification_report(y_train, predict_train))
