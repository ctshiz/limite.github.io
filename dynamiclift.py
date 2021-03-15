import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import pickle

from sklearn.linear_model import *
from sklearn.preprocessing import *
from sklearn.model_selection import *

Individual = [i for i in range(1, 26)]
Arm_Strength = [17.3, 19.3, 19.5, 19.7, 22.9, 23.1, 26.4, 26.8, 27.6, 28.1, 28.2, 28.7, 29.0, 29.6, 29.9, 29.9, 30.3, 31.3, 36.0, 39.5, 40.4, 44.3, 44.6, 50.4, 55.9]
Dynamic_Lift = [71.7, 48.3, 88.3, 75.0, 91.7, 100.0, 73.3, 65.0, 75.0, 88.3, 68.3, 96.7, 76.7, 78.3, 60.0, 71.7, 85.0, 85.0, 88.3, 100.0, 100.0, 100.0, 91.7, 100.0, 71.7]

Virginia_Tech_study = pd.DataFrame({'Individual': Individual, 'Arm_Strength': Arm_Strength, 'Dynamic_Lift': Dynamic_Lift})

x = Virginia_Tech_study[['Arm_Strength']]
y = Virginia_Tech_study[['Dynamic_Lift']]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 6)

Virginia_model = LinearRegression()
Virginia_model.fit(x_train, y_train)

#saving model to disk
pickle.dump(Virginia_model, open('model.pkl', 'wb'))

#loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[65]]))