from helper_functions import sigmoid
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class LogisticRegression():
    def __init__(self, X, y, learning_rate = 0.5, number_of_iterations = 2000, plot = True):
        self.__W = np.zeros((X.shape[0], 1))
        self.__b = 0.
        self.__X = X
        self.__y = y
        self.__m = X.shape[1]
        self.__learning_rate = learning_rate
        self.__number_of_iterations = number_of_iterations
        self.__grads = {}
        self.__plot = plot
        
    def __activate(self, X):
        activation,_ = sigmoid(np.dot(self.__W.T, X) + self.__b)
        return activation
    
    def __compute_cost(self):
        activation  = self.__activate(self.__X)
        cost = (-1 / self.__m) * (np.sum(self.__y * np.log(activation) + (1 - self.__y) * np.log(1 - activation)))
        cost = np.squeeze(cost)
        return cost
    
    def __propagate(self):
        activation = self.__activate(self.__X)
        dw = (1 / self.__m) * np.dot(self.__X, (activation - self.__y).T)
        db = (1 / self.__m) * np.sum((activation - self.__y))
        return dw, db
    
    def optimize(self):
        grads = {}
        costs = []
        for epoch in range(self.__number_of_iterations + 1):
            dw, db = self.__propagate()
            grads['dw'] = dw
            grads['db'] = db
            self.__W -= self.__learning_rate * dw
            self.__b -= self.__learning_rate * db
            cost = self.__compute_cost()
            if epoch % 100 == 0:
                Y_prediction_train = self.predict(self.__X)
                print('cost at iteration ' + str(epoch) + ' : ' + str(np.round(cost, 4)) + ' accuracy: ' + str(np.round(100 - np.mean(np.abs(Y_prediction_train - self.__y)) * 100, 4)) + '%')
                costs.append(cost)
        if self.__plot == True:
            plt.plot(np.squeeze(costs))
            plt.ylabel('Cost')
            plt.xlabel('Epochs (per tens)')
            plt.title("Learning rate =" + str(self.__learning_rate))
            plt.show()
            Y_prediction_train = self.predict(self.__X)
            print('train accuracy: ' + str(np.round(100 - np.mean(np.abs(Y_prediction_train - self.__y)) * 100, 4)) + '%')
        self.__grads = grads
        return costs
    
    def predict(self, X, y = 'empty'):
        predicted = np.zeros((1, X.shape[1]))
        activation = self.__activate(X)
        for i in range(activation.shape[1]):
            predicted[0, i] = (activation[0, i] > 0.5)
        if y != 'empty':
            self.__confusion_matrix = confusion_matrix(y.T, predicted.T)
        return predicted 
    
    def get_weights(self):
        return self.__W
    
    def get_bias(self):
        return self.__b
    
    def get_confusion_matrix(self):
        return self.__confusion_matrix
