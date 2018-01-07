#--------------------------------Imports--------------------------------#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'logistic_regression_class')))
from logistic_regression_class import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
#--------------------------Data Preprocessing--------------------------#
dataset = pd.read_csv('Pima Diabetes Dataset/diabetes.csv')
# Choosing columns
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 8].values
# Splitting data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
# Scaling features
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Getting the proper shapes
Y_train = Y_train.reshape(576, 1)
Y_test = Y_test.reshape(192, 1)
X_train = X_train.T
X_test = X_test.T
Y_train = Y_train.T
Y_test = Y_test.T 
#--------------------------Learning Process----------------------------#
learning_rates = [1, 0.5, 0.05, 0.005]
costs = []
confusion_matrix_train = []
confusion_matrix_test = []
for i in learning_rates: 
    classifier = LogisticRegression(X_train, Y_train, i, 2500, True) 
    costs.append((i, classifier.optimize()))
    Y_prediction_train = classifier.predict(X_train, Y_train)
    confusion_matrix_train.append(classifier.get_confusion_matrix())
    Y_prediction_test = classifier.predict(X_test, Y_test)
    confusion_matrix_test.append(classifier.get_confusion_matrix())
#--------------------------Comparing classifiers-----------------------#
for i in costs:
    learning_rate, cost = i
    plt.plot(np.squeeze(cost), label= "learning rate = " + str(learning_rate))
plt.ylabel('cost')
plt.xlabel('iterations')
legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
#---------------------------Calculating accuracy------------------------#
classifier_number = 0
for confusion_mat in confusion_matrix_train:
    train_accuracy = (confusion_mat[0, 0] + confusion_mat[1, 1]) / np.sum(confusion_mat)
    print("train accuracy on classifier " + str(classifier_number) + ": " + str(np.round(train_accuracy * 100, 4)) + '%')
    classifier_number += 1
classifier_number = 0
for confusion_mat in confusion_matrix_test:
    test_accuracy = (confusion_mat[0, 0] + confusion_mat[1, 1]) / np.sum(confusion_mat)
    print("test accuracy on classifier " + str(classifier_number) + ": " + str(np.round(test_accuracy * 100, 4)) + '%')
    classifier_number += 1  