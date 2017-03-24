import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

'''
    Creating dataset for clustering
'''
data = np.array([[1,3],[3,2],[4,4],[8,8],[9,7],[7,10]])
plt.style.use('ggplot')
plt.title('Data')
plt.scatter(data[:,0],data[:,1])
plt.show()

'''
    Creating the object of KMeans
'''
model = KMeans(n_clusters=2)
model.fit(X=data)
cluster_centers = model.cluster_centers_
labels = model.labels_

'''
    Displaying the clusters annd the cluster centers
'''
label_color = ["r","g"]
for i in range(len(data)):
    plt.plot(data[i][0],data[i][1],label_color[labels[i]],marker='.',markersize=10)
plt.scatter(cluster_centers[:,0],cluster_centers[:,1],marker='x',s=5,linewidth=10)
plt.show()

'''
    Now we will try to classify a new datapoint
'''
x = [[2,5]]
label = model.predict(x)
for i in range(len(data)):
    plt.plot(data[i][0],data[i][1],label_color[labels[i]],marker='.',markersize=10)
plt.scatter(x[0][0],x[0][1],c=label_color[label[0]],s=5,marker='x', linewidth=10)
plt.show()