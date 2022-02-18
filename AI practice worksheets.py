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
