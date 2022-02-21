import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import mode
import statistics as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score

breast = datasets.load_breast_cancer()

x = breast.data
y = breast.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state= 10)

#we could go ahead to fit our model with training data and predict y
#however, we want to find the k datapoints
#we would use stats, hence we import thos libraries including KNN

#we are finding the distance between k (our training set) and our predicted datapoint, which is the eucliden distance
#the closer the distance to k, the more it falls in that category
#we would use the training set of k to predict our datapoint.

#Instantiate the model
model = KNeighborsClassifier(n_neighbors= 5)

#fit the training set

model.fit(x_train,y_train)

#we would assume dp..as the datapoint we are making predictions about
#the function should return an integer that is the predicted class of dp
#k = n_neighbours

def predict_datapoint(dp, x_train, y_train, k):
    for every in x_train:
        #linalg..cals euclidean distance
        #for each data point is xtrain, find the distance from the predicted datapoint

        distance = np.linalg.norm(x_train - dp)
        # all the possible distance would come out, sort them

        #
        sort_distance = np.argsort(distance)
        #we have sorted distance of training data from dp on the x axis
        #we want to know the y of all these predicted datapoints

        classes = y_train[sort_distance[:k]]
        # obtain the classes (in Ytrain) of the datapoints with the smallest distance to pt

        MODE = st.mode(classes)
    return MODE


#now let us write a function to predict y, instead of using model.predict
#we have predicted our datapoints, now we want to predict the y of each datapoint using our testing set

PRED = []
def predict_WHY(x_test, x_train, y_train, k=5):
    for each in x_test:
        PRED.append(predict_datapoint(each, x_train, y_train, k))
    return PRED

    #We Loop over the datapoints in Xtst and store the prediction for that datapoint


pred_y = model.predict(x_test)

#the formula is simple
#but if we want to do it the math way:
#- we have to calculate the euclidean distance between our x (from the training set) and our supposed datapoints
#- do not forget to loop this under xtrain, so you can get as much datapoints as possible
#classes and mode are not compulsory..the mode will just tell you how many they are, once you match them with y
#after you get your datapoints, you can predict y from the testing set of x
#loop in the testing sets, the formula for predicting the datapoints


#check if they are the same

predict_WHY(x_test, x_train, y_train, k=5) == pred_y


#CROSS VALIDATION: LAST EXAMPLE WAS dtc VS KFOLD
#WE HAVE DONE KNN...LET US INSTANTIATE KFOLD (KNN VS KFOLD)
#Choose splits randomly

model2 = KFold(n_splits= 5, random_state= 10, shuffle= True)

#set a variable max_K

max_k = 30

#last one, we looped our training and val, kfold under dtc..this time under KNN.
#LOOP train and val indices in KFOLD

training_accuracy1 = [[] for _ in range(max_k)]
for every in range(max_k):
    #instantiate the classifier
    model = KNeighborsClassifier(n_neighbors=every)
    #loop in Kfold
    for X1_train_index, X1_test_index in model2.split(x_train):
        Xtrain, Xtest = x_train[X1_train_index], X_train[X1_test_index]
        Ytrain, Ytest = y_train[X1_train_index], Y_train[X1_test_index]
        #we want to have new Xtrains and Xtests
        #then you fit the new Xtrains and Xtests into the classifier

        model.fit(Xtrain, Ytrain)

        #whatever predictions we make, is for this classifier

        pred_y = model.predict(Xtest)

        #To find accuracy, import necessary  library
        training_accuracy = accuracy_score(y_test, pred_y )
        #store outside
        training_accuracy1[every].append(training_accuracy)
        #the appended accuracy has to be in a loop as well






