import sys
import os
import numpy as np 
from matplotlib import pyplot as plt 
from numpy import array, dot
from numpy import linalg as LA

unit_step = lambda x: -1 if x < 0 else 1

training_data = [
    [array([-3.0,-2.9,1]), -1],
    [array([0.5,8.7,1]), -1],
    [array([2.9,2.1,1]), -1],
    [array([-0.1,5.2,1]), -1],
    [array([-4.0,2.2,1]), -1],
    [array([-1.3,3.7,1]), -1],
    [array([-3.4,6.2,1]), -1],
    [array([-4.1,3.4,1]), -1],
    [array([-5.1,1.6,1]), -1],
    [array([1.9,5.1,1]), -1],
    [array([7.1,4.2,1]), 1],
    [array([-1.4,-4.3,1]), 1],
    [array([4.5,0.0,1]), 1],
    [array([6.3,1.6,1]), 1],
    [array([4.2,1.9,1]), 1],
    [array([1.4,-3.2,1]), 1],
    [array([2.4,-4.0,1]), 1],
    [array([2.5,-6.1,1]), 1],
    [array([8.4,3.7,1]), 1],
    [array([4.1,-2.2,1]), 1],
]

w = array([0.0,0.0,0.0])
eta = 0.2
flag=0
iterations=0

while flag == 0 :
    flag=1
    for i in training_data :
        x, expected_class = i[0],i[1]
        result = dot(w, x)
        error = expected_class - unit_step(result)
        if error != 0 :
            flag=0
            np.add(w,eta * error * x,out=w,casting="unsafe")
    iterations+=1

print "Iterations = ",iterations

x = np.arange(-10,10) 
for i,j in training_data :

    plt.title("Data2") 
    plt.xlabel("x axis") 
    plt.ylabel("y axis") 
    if j == -1:
        plt.plot(i[0],i[1],"og")
    else :
         plt.plot(i[0],i[1],"ob")

l = np.linspace(-10,10)
aa, bb = -w[0]/w[1], -w[2]/w[1]
plt.plot(l, aa*l+bb, 'r-')
plt.show()
