import sys
import os
import numpy as np 
from matplotlib import pyplot as plt 
from numpy import array, dot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

unit_step = lambda x: -1 if x < 0 else 1

def accuracy(Data,weight,b,c) :
	count=0.0
	for i,actual_class in Data:
		yj=0.0
		for k in range(0,len(weight)):
			yj=yj+c[k]*unit_step((dot(weight[k],i)+b[k]))
		if unit_step(yj)==actual_class :
			count+=1

	acuracy=count/len(Data)
	return acuracy

def Loadfile1(filename) :
	lines=open(filename,"rb")
	data=[]
	for line in lines :
		line=line.strip().split(',')
		del line[0]
		row=[]
		for i in range(0,len(line)) :
			if line[i].isdigit() and i< len(line)-1:
				row.append(int(line[i]))
			elif i== len(line)-1 :
				if line[i].isdigit() and int(line[i])== 2 :
					l=[array(row)]
					l.append(1)
					data.append(l)
				elif line[i].isdigit() and int(line[i])==4:
					l=[array(row)]
					l.append(-1)
					data.append(l)
			else :
				break

	return data

def Loadfile2(filename) :
	lines=open(filename,"rb")
	data=[]
	for line in lines :
		line=line.strip().split(',')
		row=[]
		for i in range(0,len(line)) :
			if i< len(line)-1:
				row.append(float(line[i]))
			elif i== len(line)-1 :
				if line[i]== 'g' :
					l=[array(row)]
					l.append(1)
					data.append(l)
				elif line[i]=='b':
					l=[array(row)]
					l.append(-1)
					data.append(l)

	return data

def voted_perceptron(data,j):
	avg_accuracy=0.0
	kf = KFold(n_splits=10)
	kf.get_n_splits(array(data))

	for train_index, test_index in kf.split(array(data)):
		n=0
		l=[]
		for i in range (0,len(data[0][0])):
			l.append(0.0)
        
		w=[array(l)]
		b=[0.0]
		c=[1]

		data1=array(data)[train_index]

		for iter in range(0,j) :
			for i in data1 :
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

		data1=array(data)[test_index]          
		avg_accuracy=avg_accuracy+accuracy(data1,w,b,c)
        
	avg_accuracy=avg_accuracy/10
	# print "voted perceptron avg=",avg_accuracy
	plt.plot(j,avg_accuracy,"ob")
	return avg_accuracy

def perceptron(data,j):
	kf = KFold(n_splits=10)
	kf.get_n_splits(array(data))
	accuracy_perceptron=0.0
	for train_index, test_index in kf.split(array(data)):
		# n=0
		l=[]
		for i in range (0,len(data[0][0])):
			l.append(0.0)
        
		w=[array(l)]
		flag=1
		b=0.0
		eta=0.2

		data1=array(data)[train_index]

		for iter in range(0,j):
			for i in data1 :
					x, expected = i[0],i[1]
					result = dot(w, x)+b
					error = expected - unit_step(result)
					if error != 0 :
						w=np.add(w,eta * error * x)
						b=b+expected
	   
		data1=array(data)[test_index]

		count=0.0
		for i in data1:
			x, expected = i[0],i[1]
			result=dot(w,x)+b
			result=expected*result
			if result>0:
				count=count+1

		accuracy_perceptron=accuracy_perceptron+count/len(data1)
	avg_accuracy_perceptron=accuracy_perceptron/10
	# print "perceptron",avg_accuracy_perceptron 
	plt.plot(j,avg_accuracy_perceptron,"og")
	return avg_accuracy_perceptron

file="cancer.txt"
data=Loadfile1(file)

Num_Epochs=[ 10, 15, 20, 25, 30, 35,40, 45, 50 ]
print "Cancer:"
avg1=[]
avg2=[]
i=0
print "voted_perceptron vanila_perceptron "
for j in Num_Epochs:

	avg1.append(voted_perceptron(data,j))   
	avg2.append(perceptron(data,j))
	print avg1[i],avg2[i]
	i=i+1
plt.title("Cancer") 
plt.xlabel("Epochs") 
plt.ylabel("Accuracy") 
plt.plot(Num_Epochs,avg1,color='b')
plt.plot(Num_Epochs,avg2,color='g')
plt.show()

file="inosphere.txt"
data=Loadfile2(file)

print "inosphere:"
avg1_p=[]
avg2_p=[]
i=0
for j in Num_Epochs:
	avg1_p.append(voted_perceptron(data,j))   
	avg2_p.append(perceptron(data,j))
	print avg1_p[i],avg2_p[i]
	i=i+1
plt.title("Inosphere") 
plt.xlabel("Epochs") 
plt.ylabel("Accuracy") 
plt.plot(Num_Epochs,avg1_p,color='b')
plt.plot(Num_Epochs,avg2_p,color='g')
plt.show()