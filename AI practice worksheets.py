#introduction to machine learning
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score

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

