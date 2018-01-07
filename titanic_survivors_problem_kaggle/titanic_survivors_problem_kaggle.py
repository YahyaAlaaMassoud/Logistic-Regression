#--------------------------------Imports--------------------------------#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'logistic_regression_class')))
from logistic_regression_class import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, Imputer, OneHotEncoder, StandardScaler
#--------------------------Data Preprocessing--------------------------#
train_set = pd.read_csv('Titanic Survivals Dataset/titanic_train.csv')
# Choosing columns
X_train = train_set.iloc[:, 2:]
X_train = X_train.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], 1)
X_train = X_train.values
# Filling missing data
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X_train[:, 2:3])
X_train[:, 2:3] = imputer.transform(X_train[:, 2:3])
# Encoding gender column
labelencoder_X = LabelEncoder()
X_train[:, 1] = labelencoder_X.fit_transform(X_train[:, 1])
# One hot encoding Passenger Class column
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X_train = onehotencoder.fit_transform(X_train).toarray()
#X_train = X_train[:, 1:]
# Scaling columns
scmm = MinMaxScaler()
X_train = scmm.fit_transform(X_train)
# Getting proper shape
X_train = X_train.T
# Y_train
Y_train = train_set.iloc[:, 1]
Y_train = np.asarray(Y_train).reshape(891,1)
Y_train = Y_train.T
#--------------------------Test Set--------------------------#
# Doing the same as Train_set with Test_set
X_test = pd.read_csv('Titanic Survivals Dataset/titanic_test.csv')
passengerIds = X_test.iloc[:, 0]
X_test = X_test.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], 1)
X_test.fillna(X_test.mean(), inplace=True)
X_test = X_test.values
labelencoder_X = LabelEncoder()
X_test[:, 1] = labelencoder_X.fit_transform(X_test[:, 1])
X_test = X_test.astype(np.float)
#X_test = onehotencoder.fit_transform(X_test).toarray()
X_test = X_test[:, 1:]
X_test = scmm.fit_transform(X_test)
X_test = X_test.T
#--------------------------Learning Process----------------------------#
learning_rates = [1, 0.5, 0.05, 0.005]
costs = []
confusion_matrix_train = []
for i in learning_rates: 
    classifier = LogisticRegression(X_train, Y_train, i, 2500, True) 
    costs.append((i, classifier.optimize()))
    Y_prediction_train = classifier.predict(X_train, Y_train)
    confusion_matrix_train.append(classifier.get_confusion_matrix())
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
#---------------------------Test Test predictions for Kaggle Submission------------------------#
classifier = LogisticRegression(X_train, Y_train, 0.5, 2000, True) 
_ = classifier.optimize()
Y_prediction_train = classifier.predict(X_test)



passengerIds = passengerIds.astype(np.int64)
passengerIds = passengerIds.reshape(418, 1)
Y_prediction_train = Y_prediction_train.T
Y_prediction_train = Y_prediction_train.astype(np.int64)
a = np.concatenate((passengerIds, Y_prediction_train), axis = 1)
df = pd.DataFrame(a,columns = ["PassengerId", "Survived"])
df.to_csv("80.35 accuracy.csv", index=False)


