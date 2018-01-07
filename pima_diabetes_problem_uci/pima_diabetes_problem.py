import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("D:/Coding/Deep Learning/My Own Library/Logistic Regression.Git")
from logistic_regression_class import LogisticRegression   
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler

dataset = pd.read_csv('Pima Diabetes Dataset/diabetes.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 8].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
Y_train = Y_train.reshape(576, 1)
Y_test = Y_test.reshape(192, 1)
X_train = X_train.T
X_test = X_test.T
Y_train = Y_train.T
Y_test = Y_test.T 
learning_rates = [1, 0.5]
costs = []
for i in learning_rates: 
    lr = LogisticRegression(X_train, Y_train, i, 2500, True) 
    costs.append((i, lr.optimize()))
    Y_prediction_train = lr.predict(X_train, Y_train)
    print("train")
    print(lr.get_confusion_matrix())
    Y_prediction_test = lr.predict(X_test, Y_test)
    print("test")
    print(lr.get_confusion_matrix())
    
    
for i in costs:
    learning_rate, cost = i
    plt.plot(np.squeeze(cost), label= "learning rate = " + str(learning_rate))
plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()




