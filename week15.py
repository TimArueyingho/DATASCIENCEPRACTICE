# we have been using KFold for cross validation, and comparing other classifiers to the KNN
# now we would use KMeans itself

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

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
    # create a new array to hold the label
    # provide the remainder for i out of k in the range of the label
    # % is remainder (for all in the range of the length of X..get the remainder
    label = np.array([all % k for all in range(len(X))])
    new_label = []
    while True:
        ##calculate the centroids of the data: euclidean distance
        # centroids: sum of all the x datapoints in a coordinate/two, sum of all the y datapoints in a coordinate/two,
        # our label would be everything looped and the remainder
        centroids = np.sum(X[label == allk]) / len(X[label == allk])
        # put centroid in array
        centroids_array = np.array(centroids, axis=0)

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









