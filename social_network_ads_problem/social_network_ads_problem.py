import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'logistic_regression_class')))
from logistic_regression_class.logistic_regression_class import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

dataset = pd.read_csv('Social Network Ads Dataset/Social_Network_Ads.csv')
X = dataset.iloc[:, [1, 2, 3]].values
Y = dataset.iloc[:, 4].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
labelencoder_X = LabelEncoder()
X_train[:, 0] = labelencoder_X.fit_transform(X_train[:, 0])
X_test[:, 0] = labelencoder_X.fit_transform(X_test[:, 0])
X_train = X_train.astype(np.float64)
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
Y_train = Y_train.reshape(300, 1)
Y_test = Y_test.reshape(100, 1)
X_train = X_train.T
X_test = X_test.T
Y_train = Y_train.T
Y_test = Y_test.T  
learning_rates = [1, 0.5, 0.05, 0.005]
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
Y_prediction_train = lr.predict(X_train, Y_train)
cm = lr.get_confusion_matrix()
Y_prediction_test = lr.predict(X_test, Y_test)
cm = lr.get_confusion_matrix()
Y_prediction_train = lr.predict(X_train, Y_train)
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
cm = confusion_matrix(Y_test.T, Y_prediction_test.T)



    
    
    
    
    
    
    
    
    
    