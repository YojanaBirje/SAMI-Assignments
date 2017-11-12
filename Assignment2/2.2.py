import sys
import os
import numpy as np 
from matplotlib import pyplot as plt 
from numpy import array, dot

unit_step = lambda x: 0 if x < 0 else 1

lines=open("inosphere.txt","rb")
data=[]
class1_data=[]
class2_data=[]
for line in lines :
        line=line.strip().split(',')
        row=[]
        for i in range(0,len(line)) :
            if i < len(line)-1:
                row.append(float(line[i]))
            elif i== len(line)-1 :
                if (line[i])=='g':
                    l=[(array(row))]
                    l.append(1)
                    data.append(l)
                elif (line[i])=='b' :
                    l=[(array(row))]
                    l.append(-1)
                    data.append(l)
            else :
                break

Num_Epochs=15
n=0
l=[]
for i in range (0,len(data[0][0])):
    l.append(0.0)
w=[array(l)]

b=[0.0]
c=[1]
for iter in range(0,Num_Epochs):
    for i in data :
        xi,yi = i[0],i[1]
        result = dot(w[n], xi)+b[n]
        if yi*result <=0 :
            n=n+1
            yii = np.ones(len(data[0][0])) + yi-1
            temp=xi*yii
            np.add(w[n-1],temp,out=temp,casting="unsafe")
            w.append(temp)
            b.append(b[n-1]+yi)
            c.append(1)
        else:
            c[n]=c[n]+1

print "Output : w   c   b"
for i in range(0,n):
    print w[i],
    print " ",c[i],
    print " ",b[i]
print "t=",n