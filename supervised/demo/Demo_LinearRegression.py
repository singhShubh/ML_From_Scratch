import pandas as pd
from matplotlib import pyplot as plt
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path+ "/../lib")
from LinearRegression import LinearRegression


df = pd.read_csv('./../data/lr_data.csv', header=None)
df = df.rename(columns={0: 'Population in 10,000s',
                        1: 'Profit in $10,000s'})
x = df.as_matrix(['Population in 10,000s'])
y = df.as_matrix(['Profit in $10,000s'])

clf = LinearRegression()
#x=LinearRegression.normalize_feature(x)
#clf.fit_gradient(x,y,alpha=0.01,plot_costJ=True)
clf.fit_closed(x,y)
clf.fit_ncg(x,y)
predicted_y = clf.predict(x)

plt.style.use('ggplot')
plt.figure(1)
plt.title('Linear Regression')
plt.scatter(x[:,-1],y)
plt.plot(x[:,-1],predicted_y,'r')
plt.xlabel('Population in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.legend(['Prediction','Training data'],loc=2)
plt.show()
print(clf.accuracy(predicted=predicted_y,actual=y))

