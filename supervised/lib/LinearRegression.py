import numpy as np
import pickle
import scipy.optimize as opt
from matplotlib import pyplot as plt


class LinearRegression:
    def __init__(self):
        self.__theta = None
        self.__X = None
        self.__Y = None
        self._trainDone = False

    @staticmethod
    def compute_cost(theta, X, Y, l=0):
        m,n = X.shape
        #-------------------------------------
        theta = np.matrix(theta)
        theta = theta.T
        #-------------------------------------
        error = X * theta - Y
        temp_theta = theta[:,1:n]
        regularization_term = l * np.sum(np.power(temp_theta, 2))
        J = (1.0 / (2.0 * m)) * (error.T * error + regularization_term)
        return J

    @staticmethod
    def compute_gradient(theta, X, Y, l= 0):
        m = X.shape[0]
        # -------------------------------------
        theta = np.matrix(theta)
        theta = theta.T
        # -------------------------------------
        error = X * theta - Y
        regularization_term = l * np.sum(theta)
        grad = (1.0 / m) * ( X.T*error + regularization_term)
        return grad.T

    @staticmethod
    def normalize_feature(X):
        n = X.shape[1]
        for i in range(n):
            X[:, i] = (X[:, i] - np.mean(X[:, i], axis=0)) / np.std(X[:, i], axis=0)
        return X

    def fit_closed(self, X, Y, l=0):
        n = X.shape[1]
        X = np.matrix(np.insert(X, 0, 1, axis=1))
        regularization_term = np.insert(np.eye(n), 0, 0, axis=1)
        regularization_term = l * np.matrix(np.insert(regularization_term, 0, 0, axis=0))
        theta = np.linalg.pinv(X.T * X + regularization_term) * X.T * Y
        self.__theta = theta
        print(self.__theta)
        self._trainDone = True

    def fit_gradient(self, X, Y, alpha, num_iters=1500, l=0, plot_costJ=False):
        X = np.matrix(np.insert(X, 0, 1, axis=1))
        Y = np.matrix(Y)
        m, n = X.shape

        theta = np.matrix(np.zeros(shape=(1,n), dtype='float'))
        costJ = np.zeros((num_iters, 1), 'float')

        for i in range(num_iters):
            gradient = LinearRegression.compute_gradient(theta, X, Y, l)
            theta = theta - alpha * gradient
            J = LinearRegression.compute_cost(theta, X, Y, l)
            costJ[i] = J

        self.__theta = theta.T
        self._trainDone = True

        if plot_costJ:
            plt.style.use('ggplot')
            plt.title('Variation in Cost with iterations:')
            plt.plot(range(num_iters), costJ, 'r')
            plt.xlabel('No. of iterations:')
            plt.ylabel('Cost (J):')
            plt.show()

#########################################################################################
    def fit_ncg(self,X,Y):
        X = np.insert(X, 0, 1, axis=1)
        n = X.shape[1]
        theta_init = np.zeros((1,n),dtype='int')
        theta = opt.fmin_tnc(func=LinearRegression.compute_cost,
                              x0=theta_init,
                              fprime=LinearRegression.compute_gradient,
                              args=(X,Y))
        self.__theta = np.matrix(theta[0]).T
        print(self.__theta)
        self._trainDone = True
#########################################################################################

    def predict(self, X):
        if self._trainDone:
            X = np.insert(X, 0, 1, axis=1)
            return X * self.__theta
        else:
            print("First train the classifier to make predictons")
            return

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

    def accuracy(self, predicted, actual):
        mean = np.mean(actual)
        acc = np.sum(np.power((predicted - mean), 2)) / np.sum(np.power((actual - mean), 2))
        return acc