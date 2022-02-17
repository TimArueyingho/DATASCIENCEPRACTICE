import numpy as np

#Imagine arrays as stacks
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

