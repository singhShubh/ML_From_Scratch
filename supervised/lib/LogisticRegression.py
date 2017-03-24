import numpy as np
import pickle
from matplotlib import pyplot as plt
import scipy.optimize as opt


class LogisticRegression:
    def __init__(self):
        self.__theta = None
        self.__trainDone = False

    @staticmethod
    def compute_cost(theta, X, Y, l=0):
        theta = np.matrix(theta)
        m, n = X.shape
        z = X * theta.T
        hypothesis = LogisticRegression.sigmoid(z)
        temp_theta = theta[1:n]
        regularization_term = (l / 2.0) * np.sum(np.power(temp_theta, 2))
        J = (-1.0 / m) * (Y.T * np.log(hypothesis) + ((1 - Y).T) * np.log(1 - hypothesis) + regularization_term)
        return J

    @staticmethod
    def compute_gradient(theta, X, Y, l=0):
        m = X.shape[0]
        theta = np.matrix(theta)
        hypothesis = LogisticRegression.sigmoid(X * theta.T)
        error = hypothesis - Y
        regularization_term = l * np.sum(theta.T)
        grad = (1.0 / m) * (X.T * error + regularization_term)
        return grad

    @staticmethod
    def sigmoid(z):
        h = np.matrix(1.0 / (1.0 + np.power(np.e, -1 * z)))
        return h

    def optm(self, X, Y):
        X = np.matrix(np.insert(X, 0, 1, axis=1))
        n = X.shape[1]
        theta = np.matrix(np.zeros(shape=(n, 1), dtype='float'))
        #print(theta.shape)
        result = opt.fmin_tnc(func=LogisticRegression.compute_cost, x0=theta,
                              fprime=LogisticRegression.compute_gradient, args=(X, Y))
        self.__theta = np.matrix(result[0]).T
        print(self.__theta)
        self._trainDone = True

    def predict(self, X):
        if self._trainDone:
            X = np.insert(X, 0, 1, axis=1)
            m = X.shape[0]
            Y = np.matrix(np.zeros(shape=(m, 1), dtype='float'))
            Y[:, 0] = LogisticRegression.sigmoid(X * self.__theta)
            res_Y = Y > 0.5 + 0
            return res_Y
        else:
            print("First train the classifier to make predictons")
            return

    def accuracy(self, predicted, actual):
        acc = 0
        n = predicted.shape[0]
        for i in range(n):
            if predicted[i] == actual[i]:
                acc += 1
        acc /= n
        return acc

    def save_model(self):
        with open("model.lr", 'wb') as f:
            pickle._dump(self, f)
        return

    def load_model(self):
        pickle_in = open("model.lr", 'rb')
        model_obj = pickle.load(pickle_in)
        self.__theta = model_obj.__theta
        self._trainDone = model_obj.__testDone
        return