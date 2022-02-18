#introduction to machine learning
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

#import the necessary libraries

#supervised ML
#Classification

#import the dataset from scikitlearn
#load the wine dataset

wine= datasets.load_wine()

#Create your dependent (y?) and independent (x) vars
x = wine.data
y = wine.target

#print(y)
#y is good for classification because the values are categorized into three sections

#divide data into training and testing sets (import tts)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 10)

#we are going to use the KNN model, import it
#we are going to instantiate it

model = KNeighborsClassifier(n_neighbors= 5)

#fit the training data into the model

model.fit(x_train,y_train)

#predict y using the test set of x
pred_y = model.predict(x_test)
#print (pred_y)

#let us evaluate our choice of model
#The confusion matrix produces a grid of positives and negatives
#The sum of the TRUE's (+ve and -ve) divided by the total number of items would give you the accuracy
#import confusion matrix

cm = confusion_matrix(y_test, pred_y)
#print (cm)

#define a function to calculate accuracy
#look at the accuracy formula

# check the formula in notes: the sum of all right diagonals in a confusion matrix divided by the total number of items
def my_accuracy(y_test, pred_y):
    cm = confusion_matrix(y_test, pred_y)
    accuracy = np.diag(cm)/cm.sum()
    return accuracy


#look at the notes
# the formula: the first item in all the arrays, divided by the sum of all the items on that array vertically
def my_precision(y_test, pred_y):
    cm = confusion_matrix(y_test, pred_y)
    precision = np.diag(cm[0])/cm[:,0].sum()
    return precision

# the formula: the first item in all the arrays, divided by the sum of all the items on that particular array horizontally
def my_recall_macro(y_test, pred_y):
    cm = confusion_matrix(y_test, pred_y)
    recall = np.diag(cm[0])/cm[0:].sum()
    return recall

#Check that your functions match those in sklearn.
#import library
my_accuracy(y_test, pred_y) == accuracy_score(y_test, pred_y)
my_precision(y_test, pred_y) == precision_score(y_test, pred_y, average = 'macro')
my_recall_macro(y_test, pred_y) == recall_score(y_test, pred_y , average = 'macro')



#Regression
#We need a dataset, this time we will create one from random numbers
dataset = np.random.default_rng(10)

#divide into dep (y) and ind (x) variables
#y = mx + b  .................linear graph
#we want to find x (x_1) and y (y_1), but we are told the values of the coefficients (m is the gradient and b the intercept)

m = 2
b = -1

#we would assume the numbers for x_1 are in a range of 1 to 10
x_1 = np.linspace(0,10,101)
#print (x_1)

#let us substitute it into the linear equation
y_1 = m*x_1 + b

#now we know our x and y
#split your data, import the library

x_1_train, x_1_test, y_1_train, y_1_test = train_test_split(x_1, y_1, test_size= 0.2, random_state= 10)

#here
x_1_reshape = x_1_train.reshape(-1,1)

#import the classifier
model_1 = LinearRegression()
model_1.fit(x_1_reshape, y_1_train)


#error: reshape something, let us check
#reshape x...do it above

gradient = model_1.coef_
intercept = model_1.intercept_

print(gradient)
print(intercept)

#Reshape the test data to have one column and
#then call predict on the regression model to get the predicted y values

x_1_test_reshape = x_1_test.reshape(-1,1)
pred_y1 = model_1.predict(x_1_test_reshape)
print(pred_y1)

#calculate the mean squared error
#the formula is sum of the difference between y and x, all squared, divided by the number of values
#remember pred_y is based on x

def MSE(y_1_test, pred_y1):
    difference = (y_1_test - pred_y1) ** 2
    mse = difference.sum()/len(difference)
    return mse

print(MSE(y_1_test, pred_y1))


#Calculate R squared
#from the notes: the formula is sum of the difference between y and x, all squared, divided by
# sum of the difference between y and the mean of y, all squared
#all of it subtracted from 1

def R_squared(y_1_test, pred_y1):
    differences = (y_1_test - pred_y1) ** 2
    sum_differences = differences.sum()
    differences_down = (y_1_test - np.mean(pred_y1)) ** 2
    sum_differences_down = differences_down.sum()
    divide_sum = sum_differences/sum_differences_down
    rsquare = 1 - divide_sum
    return rsquare

print(R_squared(y_1_test, pred_y1))


#CROSS VALIDATION FOR MODEL SELECTION TIME!!!
#we would also generate a random sample, but this time we would do it together with make_classification


X,Y = make_classification(n_samples = 2000, n_features = 10, n_classes=4, n_informative = 3, random_state=10, shuffle= True)

#split into training and test set

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state= 10, shuffle= True)

#for cross validation we would use KFOLD, so import this model
#instantiate the model with 10 folds or splits and a random state of 63

model2 = KFold(n_splits= 10, random_state= 63, shuffle= True)

# we are supposed to fit our training data, but hold on...we are CROSS VALIDATING KFOLD with DECISION TREE here!!!
#import the the decision tree model

#instantiate the model and give it a max depth of 20
#normal circumstances...this is what we should use
model3 = DecisionTreeClassifier(max_depth= 20)
max_depth= 20

#now we have two instantiated models
#next we are supposed to split into training and testing (this time validating) data
#BUT: max depth in decision tree...the tree would continue to split until it is pure..it would happen 20 times here
#for each max depth we want to fit our model and make predictions
#summary: we would split, fit and predict through each max_depth
#for 2 items in the split model of X..define new values

#for all in the max depths of a DTC, instantiate the DTC model..then for the two items in the KFmodel, fit DTC and make
#predictions for #DTC

use_accuracies = [[] for _ in range(max_depth)]
use_accuracies2 = [[] for _ in range(max_depth)]

for every in range(max_depth):
    model3 = DecisionTreeClassifier(max_depth= every + 1)

    for X1_train_index, X1_test_index in model2.split(X_train):
        Xtrain, Xtest = X_train[X1_train_index], X_train[X1_test_index]
        Ytrain, Ytest = Y_train[X1_train_index], Y_train[X1_test_index]

        model3.fit(Xtrain, Ytrain)  #This is for DTC

        # This is for dtc
        pred_train = model3.predict(Xtrain)
        pred_test = model3.predict(Xtest)

        # find accuracy: remember you have to find the cm first
        def new_accuracy(Ytest, pred_test):
            cm = confusion_matrix(Ytest, pred_test)
            acc = np.diag(cm).sum() / cm.sum()

        #We can also use the precision, accuracy metric
        train_accuracy = accuracy_score(Ytest, pred_test)
        #if you want to print it outside the for loop, you have to create a list and append
        #the empty list would be above.. you can place it now
        use_accuracies[every].append(accuracy_score(Ytest, pred_test))
        #the [every] is for the accuracies to be in a loop

        #you can do the same for the training set
        train_accuracy2 = accuracy_score(Ytrain, pred_train)
        # if you want to print it outside the for loop, you have to create a list and append
        # the empty list would be above.. you can place it now
        use_accuracies2[every].append(accuracy_score(Ytrain, pred_train))
        # the [every] is for the accuracies to be in a loop


#when we run the code, it would tell us that the list is out of range (the loop is never ending)
#hence , we have to edit the list from use_accuracies []

#Calculate the mean and standard deviation for training
#and validation/testing accuracies for each depth across splits
use_accuracy_mean = np.mean(use_accuracies, axis=1)
use_accuracy_std = np.std(use_accuracies, axis=1)
use_accuracy_std2 = np.std(use_accuracies2, axis=0)
use_accuracy_mean2 = np.mean(use_accuracies2, axis=0)




