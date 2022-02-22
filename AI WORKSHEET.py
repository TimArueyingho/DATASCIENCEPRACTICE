# Import suitable packages, load the dataset, and save data and targets into variables X and Y
import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score

breast = datasets.load_breast_cancer()
x = breast.data
y = breast.target

#we want to write our own implementation of naive bayes
#we have to use the statistics module
#normally, we would split our dataset into the training and test sets, before instantiating the classifier

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state= 10, shuffle= True)



#no need to importa model, instantiate, fit and predict, we have to divide our dataset into classes
#then use those classes to calculate the prior probability
#it is the prior probability that would be used for prediction

#Separate the training set into classes, so you have one set of data for each class
#classify x,y datapoints of your training set
##TODO##

class_1 = x_train[y_train == 0]
class_2 = x_train[y_train == 1]

# Calculate the means and standard deviations for each class, for each feature.

class_1mean = np.mean(class_1, axis = 1)
class_2mean = np.mean(class_2, axis = 1)
class_1std = np.std(class_1, axis = 1)
class_2std = np.std(class_2, axis = 1)

#in naive bayes, you have to calculate the prior probability
#because that is what you will use for your prediction

#the prior probability formula: probability of class2 divided by class 1,
#multiplied by the probability of class1, all divided by the probability of class 2

#probability = event/total number of outcomes  (the length of selected class/the length of xtraining set)
#x training set, because we want to predict y
#find the probabilities of class 1 and class 2

prob_class1 = len(class_1)/len(x_train)
prob_class2 = len(class_2)/len(x_train)

prior_probability = ((prob_class2/prob_class1) * prob_class1)/prob_class2


# Calculate the log-likelihood of each class for each datapoint in the validation set
# log of probabilities + sum of the normal logpdfs

ll = np.log(prob_class1) + np.sum(norm.logpdf(x_test, loc=class_1mean, scale=class_1std), axis=1)

ll2 = np.log(prob_class2) +np.sum(norm.logpdf(x_test, loc= class_2mean, scale= class_2std), axis=1)


#If we want to check results, now we can import the NaiveBayes model and instantiate it


model = GaussianNB(var_smoothing= 0.0)
model.fit(x_train,y_train)
pred_y_model = model.predict(x_test)
pred_y_class1 = model.predict(class_1)
pred_y_class2  = model.predict(class_2)

print(pred_y_model, pred_y_class2, pred_y_class1)



#CROSS VALIDATION: AGAIN!!!! kfold is always used
#if K-fold is always used for cross validation, you compare the classifier or classifiers under kfold
#THIS TIME: KNN AND GAUSSIAN NAIVE BAYES

#import KNN and KFold

max_depth = 30
model = GaussianNB(var_smoothing= 0.0)
model2 = KFold(n_splits= 5, random_state= 10, shuffle= True)

#we can fit in our training data and make predictions like above

PREDICTION = []
SAVE_ACCURACY = [ [] for all in range (max_depth)]

for every in range(max_depth):
    model = GaussianNB(var_smoothing= 0.0, priors= max_depth)

    for x_index, y_index, in model2.split(x_train):
        X_TRAIN, X_TEST = x_train[x_index], x_train[y_index]
        Y_TRAIN, Y_TEST = x_train[x_index], x_train[y_index]

        #fit into GNB and predict
        model.fit(X_TRAIN,Y_TRAIN)
        PRED_Y = model.predict(X_TEST)

        PREDICTION[every].append(PRED_Y)

        #test for accuracies
        MY_ACCURACY = accuracy_score(Y_TEST, PRED_Y)
        SAVE_ACCURACY[every].append(MY_ACCURACY)


#Calculate the mean of the accuracies
print(np.mean(SAVE_ACCURACY))



#How to compare two different classifiers
#you would have to instantiate both models, fit them into classifiers and make predictions
#use the original train test split..

#let us compare GNB and KNN

model = GaussianNB(var_smoothing= 0.0)
model3 = KNeighborsClassifier(n_neighbors= 5)

model.fit(x_train, y_train)
model3.fit(x_train, y_train)

prediction = model.predict(x_test)
prediction2 = model3.predict(x_test)

print(prediction, prediction2)