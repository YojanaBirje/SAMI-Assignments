import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

with open("Data_SVM.csv") as f:
    lis=[line.split() for line in f]
x=[]
y=[]
Y=[]
arr=[]
lis.pop(0)
for i in lis:
  l=i[0].split(',');
  x.append(l[0])
  y.append(l[1])
  temp=[]
  temp.append(float(l[0]))
  temp.append(float(l[1]))
  arr.append(temp)
  Y.append(int(l[2]))
# print np.array(arr)
X=np.array(arr)
Y=np.array(Y)

x1=[]
y1=[]
x2=[]
y2=[]
for i in range(0,len(x)):
  if int(Y[i])==-1:
    x1.append(x[i])
    y1.append(y[i])
  else:
    x2.append(x[i])
    y2.append(y[i])

plt.scatter(x1,y1,color='red')
plt.scatter(x2,y2,color='blue')
plt.show()
# kernel=poly
c=[1,2,5,20,50,10000,20000]
d=[2]
max_avg_acc=0.0
for i in c:
  for j in d:

    for k in range(0,30):
      kf = KFold(n_splits=10,shuffle=True)
      kf.get_n_splits(X)
      # avg=0.0
      # total_avg_acc=0.0
      # max_avg_acc=0.0
      acc_list=[]
      for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        clf = svm.SVC(kernel='poly',C=i,degree=j)
        clf.fit(X_train, Y_train)
        var=clf.score(X_test,Y_test)
        acc_list.append(var)
        
      #   count=0
      #   avg_acc=0.0
      #   for l in range(0,len(X_test)):
      #       sample=np.array(X_test[l])
      #       sample=sample.reshape(1,-1)
      #       if clf.predict(sample) == Y_test[l]:
      #         count=count+1
      #   avg_acc=count*1.0/len(X_test)
      #   avg=avg+avg_acc
      # total_avg_acc=avg*1.0/10
      # if total_avg_acc>max_avg_acc:
      #   max_avg_acc=total_avg_acc
      #   c1=i
      #   d1=j
      # print 'c=',i, ' d=',j,' average acc=',total_avg_acc
      m=1.0*np.mean(acc_list)
      if m>max_avg_acc:
        max_avg_acc=m
        c1=i
        d1=j
      std=np.std(acc_list)
      print "mean,std = ",m,std

# final train using best combination of c and d
print "Best combination of c and d is ",c1," ",d1
print "accuracy = ",max_avg_acc
clf = svm.SVC(kernel='poly',C=c1,degree=d1)
clf.fit(X, Y)
plt.clf()
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,facecolors='none', color='green')
        # plt.scatter(X[:, 0], X[:, 1],c=Y)
plt.scatter(x1, y1,color='red')
plt.scatter(x2, y2,color='blue')
x_min = -1
x_max = 1
y_min = -1
y_max = 1

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
Z = Z.reshape(XX.shape)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],levels=[-.5, 0, .5])

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

# score=cross_val_score(clf,X,Y,cv=10)
# acc=(score.mean,score.std()*2)
# print "acc= ",acc[0]

# kernel=rbf
# c=[2,5,20,50,10000]
max_avg_acc=0.0
sigma=[2,3]
for i in c:
  for j in sigma:

    for k in range(0,30):
      kf = KFold(n_splits=10,shuffle=True)
      kf.get_n_splits(X)
      # avg=0.0
      # total_avg_acc=0.0
      # max_avg_acc=0.0
      acc_list=[]
      for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        clf = svm.SVC(kernel='rbf', gamma=j,C=i)
        clf.fit(X_train, Y_train)
        var=clf.score(X_test,Y_test)
        acc_list.append(var)
        plt.clf()
      #   count=0
      #   avg_acc=0.0
      #   for l in range(0,len(X_test)):
      #       sample=np.array(X_test[l])
      #       sample=sample.reshape(1,-1)
      #       if clf.predict(sample) == Y_test[l]:
      #         count=count+1
      #   avg_acc=count*1.0/len(X_test)
      #   avg=avg+avg_acc
      # total_avg_acc=avg*1.0/10
      # if total_avg_acc>max_avg_acc:
      #   max_avg_acc=total_avg_acc
      #   c1=i
      #   sigma1=j
      # print 'c=',i, ' sigma=',j,' average acc=',total_avg_acc
  
      m=1.0*np.mean(acc_list)
      if m>max_avg_acc:
        max_avg_acc=m
        c1=i
        sigma1=j
      std=np.std(acc_list)
      print "mean,std = ",m,std
      
print "Best combination of c and sigma ",c1," ",sigma1
print "acuuracy = ",max_avg_acc
clf = svm.SVC(kernel='rbf', gamma=sigma1,C=c1)
clf.fit(X, Y)
plt.clf()
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,facecolors='none', color='green')
        # plt.scatter(X[:, 0], X[:, 1],c=Y)
plt.scatter(x1, y1,color='red')
plt.scatter(x2, y2,color='blue')
x_min = -1
x_max = 1
y_min = -1
y_max = 1

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
Z = Z.reshape(XX.shape)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],levels=[-.5, 0, .5])

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
