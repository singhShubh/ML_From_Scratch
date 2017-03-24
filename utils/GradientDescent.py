import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import optimize as opt
def compute_cost(x,y,theta):
    X = np.matrix(x)
    Y = np.matrix(y)
    theta = np.matrix(theta)
    error = X * theta - Y
    m = len(y)
    J = (1 / (2 * m)) * (error.T * error)
    return J

def gradientDescent(x,y,theta,alpha,num_iters):
    X = np.matrix(x)
    Y = np.matrix(y)
    theta = np.matrix(theta)
    m=len(y)
    cost = np.zeros((num_iters,1),'float')
    for i in range(num_iters):
        error = X*theta - Y
        gradient = (X.T*error)/m
        theta = theta - alpha*gradient
        J = compute_cost(X,Y,theta)
        cost[i]=J
    return theta,cost
#
# x = np.linspace(0,4,30)
# y = (x-2)**2 + 5
# plt.plot(x,y,'r')
# plt.scatter(x,y)
# plt.show()
# print(x)

df = pd.read_csv('housing_data.csv', header=None)
df = df.rename(columns={0:'Population in 10,000s',
                        1: 'Profit in $10,000s'})
temp= df.as_matrix(['Population in 10,000s'])

# Feature normalization
# With feature normatization, rate of convergence is way high
# for normalized featues: theta = [[5.8391334],[4.59303983]]
# without normalization: theta = [[-3.63029144],[1.16636235]]
num_features = temp.shape[1]
for i in range(num_features):
    print(i)
    temp[:,i] = (temp[:,i]-np.mean(temp[:,i], axis=0))/np.std(temp[:,i], axis=0)
m,n = temp.shape
n+=1
x= np.ones((m,n),'float')
x[:,1:]=temp
y= df.as_matrix(['Profit in $10,000s'])
theta = np.zeros((2,1),'float')
alpha = 0.01
num_iters = 1500

theta,cost = gradientDescent(x,y,theta,alpha,num_iters)

print('Theta =',theta)


plt.figure(1)
plt.plot(range(num_iters),cost)
plt.figure(2)
plt.scatter(x[:,-1],y)
val = x*theta
plt.plot(x[:,-1],val,'r')
plt.show()
