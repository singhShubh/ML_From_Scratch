import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt



'''
    Creating dataset for clustering
'''

plt.style.use('ggplot')
data = np.array([[1,3],[3,2],[4,4],[8,8],[9,7],[7,10]])


clf = KMeans(n_clusters=2)
clf.fit(X=data)
cluster_centers=clf.cluster_centers_
labels=clf.labels_
label_color = ["r.","g."]

for i in range(len(data)):
    plt.plot(data[i][0],data[i][1],label_color[labels[i]], markersize=10)
plt.scatter(cluster_centers[:,0], cluster_centers[:,1],s=10,marker='x',linewidths=10)
plt.show()