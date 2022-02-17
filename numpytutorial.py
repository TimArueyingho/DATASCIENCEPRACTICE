import numpy as np
import random

#Imagine arrays as stacks
#0/1/y/x/column/row
#1D
a = np.array([1, 2, 3, 4, 5, 6])
print(a)

#2D
b = np.array([[0., 0., 0.],[1., 1., 1.]])
print (b.shape)
#if you stack it, you will see that y axes (axis = 0) has a length of 2 and x axes (axis = 1), a length of three
#(y,x) (0,1) axis

#an array full of zeros
c = np.zeros(7)
print (c)

#an array full of ones
d = np.ones(6)
print(d)

#a range of elements: from zero till the last index
e = np.arange(11)
print (e)

#a range of elements: with spaced intervals
#(x,y,z) start from x till index of y, in a range of z
f = np.arange(0, 5, 2)
print (f)

#np.linspace() to create an array with values that are spaced linearly in a specified interval:
#(it would give you the values between x and y, the number of times you have equated in num)
g = np.linspace(0,10, num=51)
print (g)

array_example = np.array([[[0, 1, 2, 3],[4, 5, 6, 7]],  [[0, 1, 2, 3],[4, 5, 6, 7]],[[0 ,1 ,2, 3],[4, 5, 6, 7]]])

print (array_example.shape)
#(dimension, axis 0 and axis 1)

print (array_example.size)
#number of elements

#when reshaping, remember the number of elements inside. 2D will remain 2D, 1D,1D ETC
h = array_example.reshape(3, 1,8 )
print (h)

#if you want to reshape to another dimension, you would need to add an extra axis

#you can stack arrays, but they need to have the same size and shape
a1 = np.array([[1, 1],
               [2, 2]])

a2 = np.array([[3, 3],
               [4, 4]])

#vstack, stacks vertically...hstack..stacks horizontally

x = np.vstack((a1, a2))
y = np.hstack((a1, a2))

print (x)
print (y)


#we can create an array full of values of our own choice
#we want the array to be full of 6, as 4 of axis 0 and 3 of axis 1
fulls = np.full((4,3), 6)
print(fulls)

#we can add a particular number, diagonally
#(axis 0= 3, axis 1 = 7, the diagonals would be 1...(x,y,z)...z states the index from which the diagonal would start
identity = np.eye(3,7,1)
print(identity)

#generate random numbers in an array with the specified shape
randoms = np.random.rand(4,3)
print(randoms)

# Create an array of random integers
#we specify the highest number of the integer we want and the shape of the array (axis 0/y = 4, axis 1/x = 3
randInts = np.random.randint(10, size = (4,3)) # The syntax requires we specify␣
print(randInts)


#let us slice the array, generate the first two columns
print (randInts[0:2])

#let us slice the array, generate the last two columns
print(randInts[-2:])

#let us slice it vertically, and pull out the last two columns instead of rows
#(axis 0/y, axis1/x...0..by putting the comma , we are separating them into axes
#we need index 1 till the end.. zero is the first column we are ignoring

print (randInts[:,1:])

#we could do some maths on numpy arrays #they must be the same shape
x = np.array([[53, 74],
              [2, 14]])
y = np.array([[31.3, 0.8],[12.7, 8.1]])

print (x + y)
print (x - y)
print(np.sqrt(x)) #square root of x

#calculate the sum of the values in just the column (axis = 0)
print(np.sum(x, axis = 0))

#calculate the sum of the values in just the row (axis = 1)
print (np.sum(x, axis = 1))

