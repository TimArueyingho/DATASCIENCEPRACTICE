# we have been using KFold for cross validation, and comparing other classifiers to the KNN
# now we would use KMeans itself

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.mixture import GaussianMixture

# usually, we could set data and target for our datasets
# or we could make a classification, as in x, y
# let us make a classification

x, y = make_blobs(centers=3, n_samples=100, cluster_std=2, random_state=100)

# let us plot x against y....
fig, ax = plt.subplots()
ax.scatter(x[:, 0], x[:, 1])


#:,0 all of axis 0..... :,1....all of axis 1



# Another one..we shall generate random numbers here
# Implement a function kmeans that takes a value  𝑘  and the data  𝑋 ,
# clusters the data, and returns the centroids and the labels of the data

def kmeans(k, X):
    data_set = default_rng()
    # randomly assign labels to the data (either randint, or just generate numbers..using k)
    #generate a range of numbers that would connect k and X and that would be the data set
    # create a new array to hold the label
    # provide the remainder for i out of k in the range of the label
    # % is remainder (for all in the range of the length of X..get the remainder
    label = np.array([all % k for all in range(len(X))])
    new_label = []
    while True:
        ##calculate the centroids of the data: euclidean distance
        # centroids: sum of all the x datapoints in a coordinate/two, sum of all the y datapoints in a coordinate/two,
        # our label would be everything looped and the remainder
        #the label is our data
        centroids = np.sum(X[label == allk]) / len(X[label == allk])
        # put centroid in array
        centroids_array = np.array(centroids, axis=0)
        #centroids array is y, we need to find all the x's

        # enumerate our X (it would be listed one by one, the i..removes the bracket)
        for i, x in enumerate(X):
            #all the centroids would be listed one by one
            #we want our enumerated number X to be removed from a... the centroids array we generated are essentially y
            for a in centroids_array:
                euclidean = np.linalg.norm(x - a) ** 2
                #array the euclidean
                euclidean_array = np.array(euclidean)
            #let us get the minimum elements from the list of the array and store it as a new label.
            #the minimum is the perfect point
            #to store new label..open list outside the loop
            new_labels[i] = np.argmin(list(euclidean_array))
            new_label.append(new_label[i])

            break

            #the variables we generated would be equal to our new labels
            label[:] = new_label


#MAJOR KMEANS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#elbow plots are used to predict the number of clusters
#we can use inertia to calc elbow plots

#we have to assume the value of k
#we find the euclidean distance/manhattan distance/inertia of each datapoint to k
#inertia simple because after assuming k, we just fit into the model and find the inertia of the model
#a plot of the sum of all the distances against the values of k would give us an elbow plot

#get dataset, divide into x,y, tts, instantiate, fit and prdict

#import kmeans

value = []
K = 10
for k in range(1, K + 1):
    model = KMeans(n_clusters= k)
    model.fit(x)
    inertia = model.inertia_

    # Store the value of the inertia for this value of k
    value.append(inertia)



# Plot the elbow
plt.figure()
plt.plot(range(1, K+1), value, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('The elbow method showing the optimal k')
#plt.show()


#let us play around with the Iris dataset

iris = datasets.load_iris()
#saved data into a variable
datax = iris.data

#in this dataset, there are 3types of iris flowers...so there may be three clusters
#the aim is to print out the centroids and visualize clusters

model2 = KMeans(n_clusters= 3)
model2.fit(datax)


# Plot the elbow
## Make a scatter plot of the data on the first two axes..............
#the two axes of the data we have, plot it


plt.figure()
plt.scatter(datax[:,0],datax[:,1], c=model2.labels_, cmap='rainbow')
#plt.show()

#for this Iris dataset, we assumed K, just like the last one that we assumed K and used inertia to find
#the ideal k using an elbow plot

#generate an elbow plot
#remember we set our number of clusters to 3
#but we will assume K is 10 again
KK = 10
value_of_this_inertia = []
for all in range(1, KK + 1):
    model2 = KMeans(n_clusters=all)
    model2.fit(datax)

    inertium = model2.inertia_
    value_of_this_inertia.append(inertium)


# Plot the elbow
plt.figure()
plt.plot(range(1, KK+1), value_of_this_inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('The elbow method showing the optimal kp')

#We can generate what is called a dendrogram...a tree like cluster
#import relevant module

linked = linkage(datax, 'single')
labelList = range(len(datax))
plt.figure(figsize=(10, 7))
dendrogram(linked,labels=labelList)
plt.show()


#Another model to try: the Gaussian Mixture models
#import module

model3 = GaussianMixture(n_components= 3)
model3.fit(datax)

print(model3.means_)
print(model3.covariances_)