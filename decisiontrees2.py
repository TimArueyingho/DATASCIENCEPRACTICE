from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib.pylab import rcParams
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.tree import export_text
from sklearn import tree

#load dataset
breast_cancer = load_breast_cancer()

#separate data into dep and indep var
X = breast_cancer.data
y = breast_cancer.target

#split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 10)

#instantiate decision tree classifier
model = DecisionTreeClassifier(criterion= 'entropy')

#fit training data
model.fit(X_train,y_train)

#make predictions
pred_y = model.predict(X_test)
print(pred_y)

#calculate accuracy
my_accuracy = accuracy_score(y_test, pred_y)
print(my_accuracy)

#One of the advantages of decision tree classifiers is that they are very interpretable.
# This is because the tree corresponds to a set of if-then rules identifying the class
# of an input given various constraints on the features.
# The structure of these rules can be seen by visualising the tree.
# You can use the following commands to extract a text-based representation.
#import text

features = breast_cancer['feature_names'].tolist()
r = export_text(model,feature_names=features)
print(r)

#if you want to use a graph
rcParams['figure.figsize'] = [20, 10]
classes=breast_cancer.target_names
tree.plot_tree(model,feature_names=features,class_names=classes,fontsize=10)


#we can also instantiate the DTC using a gini impurity
# The Gini impurity can be used as an alternative to entropy when choosing on which attributes to split.

model2 = DecisionTreeClassifier(criterion= 'gini')
model2.fit(X_train, y_train)
pred_y2 = model2.predict(X_test)
print(pred_y2)

#There are various constraints that we can put on decision tree learning
# which can affect overfitting and underfitting of the data.
# Perhaps the simplest is restricting the depth of the tree
#removing constraints to prevent overfitting and underfitting


#we will do the same as the beginning, but this time we will set a max_depth and compare accuracy
model3 = DecisionTreeClassifier(criterion= 'entropy', max_depth= 2, random_state= 10)

#fit training data
model3.fit(X_train,y_train)

#make predictions
pred_y3= model3.predict(X_test)
print(pred_y3)

#calculate accuracy
my_accuracy2 = accuracy_score(y_test, pred_y3)
print(my_accuracy2)

#accuracy is less

#another method for constraining
#Another approach is to restrict the number of leaf nodes.
#In this case the tree is grown in a best first manner where best nodes
# are defined using relative reduction in impurity.

model4 = DecisionTreeClassifier(criterion= 'entropy', max_depth= 2, random_state= 10, max_leaf_nodes= 5)

#fit training data
model4.fit(X_train,y_train)

#make predictions
pred_y4= model4.predict(X_test)
print(pred_y4)

#calculate accuracy
my_accuracy3 = accuracy_score(y_test, pred_y4)
print(my_accuracy3)


#all od these are for generalization...reducing constraints
#apart from classification, decision trees can be used for regression

