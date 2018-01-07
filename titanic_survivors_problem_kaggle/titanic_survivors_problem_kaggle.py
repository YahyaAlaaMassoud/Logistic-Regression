import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("D:/Coding/Deep Learning/My Own Library/Logistic Regression.Git")
from logistic_regression_class import LogisticRegression   
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, Imputer, OneHotEncoder, StandardScaler

train_set = pd.read_csv('Titanic Survivals Dataset/titanic_train.csv')

X_train = train_set.iloc[:, 2:]
X_train = X_train.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], 1)
X_train = X_train.values

imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X_train[:, 2:3])
X_train[:, 2:3] = imputer.transform(X_train[:, 2:3])

labelencoder_X = LabelEncoder()
X_train[:, 1] = labelencoder_X.fit_transform(X_train[:, 1])

onehotencoder = OneHotEncoder(categorical_features = [0])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_train = X_train[:, 1:]

sc = StandardScaler()
scmm = MinMaxScaler()
X_train = scmm.fit_transform(X_train)

X_train = X_train.T


X_test = pd.read_csv('Titanic Survivals Dataset/titanic_test.csv')
passengerIds = X_test.iloc[:, 0]
X_test = X_test.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], 1)
X_test.fillna(X_test.mean(), inplace=True)
X_test = X_test.values

labelencoder_X = LabelEncoder()
X_test[:, 1] = labelencoder_X.fit_transform(X_test[:, 1])

X_test = X_test.astype(np.float)

X_test = onehotencoder.fit_transform(X_test).toarray()
X_test = X_test[:, 1:]

X_test = scmm.fit_transform(X_test)

X_test = X_test.T

Y_train = train_set.iloc[:, 1]
Y_train = np.asarray(Y_train).reshape(891,1)
Y_train = Y_train.T

learning_rates = [0.5, 0.05, 0.005]
costs = []
for i in learning_rates: 
    lr = LogisticRegression(X_train, Y_train, i, 5000, True) 
    costs.append((i, lr.optimize()))
    Y_prediction_train = lr.predict(X_train, Y_train)
    print("train")
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

lr = LogisticRegression(X_train, Y_train, 0.5, 2000, True) 
_ = lr.optimize()
Y_prediction_train = lr.predict(X_test)



passengerIds = passengerIds.astype(np.int64)
passengerIds = passengerIds.reshape(418, 1)
Y_prediction_train = Y_prediction_train.T
Y_prediction_train = Y_prediction_train.astype(np.int64)
a = np.concatenate((passengerIds, Y_prediction_train), axis = 1)
df = pd.DataFrame(a,columns = ["PassengerId", "Survived"])
df.to_csv("80.35 accuracy.csv", index=False)


