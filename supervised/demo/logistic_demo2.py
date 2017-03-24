import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path+ "/../lib")
from LogisticRegression import LogisticRegression


df = pd.read_csv('../data/classification_1.txt', header=None)
df = df.rename(columns={0:'Exam1',
                   1:'Exam2',
                   2:'Label'})
print(df.head())
y = df.as_matrix(['Label'])
x = df.as_matrix(['Exam1','Exam2'])
pass_df =df[df['Label']==1]
fail_df =df[df['Label']==0]
plt.style.use('ggplot')
plt.figure(1)
plt.scatter(pass_df['Exam1'],pass_df['Exam2'],c='g')
plt.scatter(fail_df['Exam1'],fail_df['Exam2'],c='r')
plt.legend(['Pass','Fail'],loc=1)
plt.show()

clf = LogisticRegression()
clf.optm(x,y)
predicted_y = clf.predict(x)
df['predicted_label'] = predicted_y
plt.scatter(pass_df['Exam1'],pass_df['Exam2'],c='g')
plt.scatter(fail_df['Exam1'],fail_df['Exam2'],c='r')
plt.legend(['Pass','Fail'],loc=1)
correctly_predicted = df[df['Label'] == df['']]
print(clf.accuracy(predicted_y, y))
