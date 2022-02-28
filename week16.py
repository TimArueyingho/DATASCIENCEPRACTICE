#Classification using single layer perceptron model (ANN)

#our data we shall create, then we would train the model and classify
import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

x,y = make_classification(n_features=2, n_redundant=0, n_informative=1, random_state=1,n_clusters_per_class=1)

#let us visualize this data
fig, ax = plt.subplots();
ax.scatter(x[:,0],x[:,1],c=y, cmap='rainbow')

#Notice that the classes are linearly separable
#We can now fit a single layer neural network to the data.
#This data is what will be multiplied by weights, added to the bias, and passed through the activation function
#threshold function...for classification

#import perceptron
#instantiate the model and train the model with data

model =Perceptron(alpha=1, max_iter=1000, fit_intercept=True)
model.fit(x,y)

#usually we would predict y, using the xtest...but we didnt even split into training and testing sets

#let us create a line to separate the data
#generate numbers between -3 and 3...
xx= np.linspace(-3,3, 101)

#y = mxx + b
#m== gradient
m = model.coef_[0][0]/model.coef_[0][1]

c = model.intercept_/model.coef_[0][1]
ys = m*xx+c
ax.plot(xx, ys)
fig


#we can see where the line divides the clusters into two
#shaaa...let us go ahead an make our normal prediction

pred_y = model.predict(x)
print(pred_y)

#let us check the accuracy of our prediction
#import accuracy score

my_accuracy = f'This is the accuracy: accuracy_score{(y, pred_y)}'
print(my_accuracy)

#For SNN...IT IS LINEARLY SEPARABLE
#Let us consider data thats not linearly separble = multi layered network

#make moons...data points would be moon shaped
#import make_moons from sk.learn

x1,y1 = make_moons(n_samples= 50, shuffle= True, random_state= 1)

#let us visualize it
fig1, ax1 = plt.subplots()
ax1.scatter(x1[:,0],x1[:,1],c=y1, cmap='rainbow')

#from the charts, we cannot use a straight line to separate the moons
#a perceptron can only fit a single line...let us see what happens when we attempt to use it here
#some of the data are mixed in the curve

model2=Perceptron(alpha=1, max_iter=1000)
model2.fit(x1, y1)
pred_y2 = model2.predict(x1)

#plot a graph of these predictions
fig2, ax2 = plt.subplots()
ax2.scatter(x1[:,0],x1[:,1],c=pred_y2, cmap='rainbow')
fig2


#A single layer perceptron can only fit a single line to data,
#and this dataset can't be separated with a single line. Hence, we can't achieve perfect performance

#hence, we need to use a multi layered NN
#IMPORT THE MLClassifier

#instantiate the model, fit training data and predict

model3=MLPClassifier(alpha=1,hidden_layer_sizes=(10,10,10,10), max_iter=1000)
model3.fit(x1,y1)
pred3 = model3.predict(x1)

#let us view the chart of this prediction
fig3, ax3 = plt.subplots()
ax3.scatter(x1[:,0],x1[:,1],c=pred3, cmap='rainbow')
fig2


#pure curves/moons
#we used 4 hidden layers with size 10 each/10 neurons each
#change hidden layers and observe chart

#let us try the same with circle like data
#import make_circles data

x2, y2 = make_circles(n_samples= 100, shuffle= True, random_state= 10)

#we will use the MLclassifier
#instantiate it, and fit the data inside

model4=MLPClassifier(alpha=1,hidden_layer_sizes=(10,100,100,10), max_iter=1000)
model4.fit(x2,y2)
pred4 = model4.predict(x2)

#let us view the chart of this prediction
fig4, ax4 = plt.subplots()
ax4.scatter(x2[:,0],x2[:,1],c=pred4, cmap='rainbow')
fig4
plt.show()

#just one class is showinf with our hidden layers (10,10,10,10)..lets do 10,100,100,100