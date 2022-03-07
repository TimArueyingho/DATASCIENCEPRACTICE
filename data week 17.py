import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

#Import the diabetes dataset

diabetes = datasets.load_diabetes()

#Clustering
#Create a copy of the dataset and Discard the target variable.
x = diabetes.data
y = diabetes.target

#discard y

#Cluster the dataset using k means clustering.
model = KMeans(n_clusters= 2)
model.fit(x)

#Try different values of k.
#so we must have a range for k

K =10

#Using the elbow method try to find best value of k.
#remember we have to use inertia

get_inertia = []
for all in range(1, K + 1):
    model = KMeans(n_clusters=all)
    model.fit(x)

    inertium = model.inertia_
    get_inertia.append(inertium)

# Plot the elbow
plt.figure()
plt.plot(range(1, K+1), get_inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('The elbow method showing the optimal kp')


#Using the best value of K from this analysis, Implement GMM to cluster the data
#import GaussianMixture

model2 = GaussianMixture(n_components= 7)
model2.fit(x)
pred_y = model.predict(x)
print(pred_y)


plt.figure()
plt.scatter(x[:,0],x[:,1], c=model2.predict(x), cmap='rainbow')
plt.show()

