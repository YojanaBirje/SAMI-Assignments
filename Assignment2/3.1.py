import numpy as np
import matplotlib.pyplot as plt
from numpy import array, dot
from numpy import linalg as LA

unit_step = lambda x: 0 if x < 0 else 1

def fisher(dataset):

    class1_data=[]
    class2_data=[]
    
    for i,actual_class in dataset :
        if actual_class==1 :
            class1_data.append([float(i[0]),float(i[1])])
        else :
            class2_data.append([float(i[0]),float(i[1])])

    class1_data=array(class1_data)
    class2_data=array(class2_data)

    mean1 = class1_data.mean(axis = 0).T
    mean2 = class2_data.mean(axis = 0).T

    m,n = class1_data.shape
    diff1 = class1_data - np.array(list(mean1)*m).reshape(m,n)

    m,n = class2_data.shape
    diff2 = class2_data - np.array(list(mean2)*m).reshape(m,n)

    diff = np.concatenate([diff1, diff2])

    m, n = diff.shape
    withinClass = np.zeros((n,n))
    diff = np.matrix(diff)
    
    for i in xrange(m):
        withinClass += np.dot(diff[i,:].T, diff[i,:])

    fisher_vec = np.dot(np.linalg.inv(withinClass), (mean1 - mean2))

    print "Fisher's linear classifier :",fisher_vec

    x = np.arange(-4,10) 
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    space = np.linspace(-10,10)

    norm_w=LA.norm(fisher_vec)
    points_1=[]
    for i in class1_data :
        proj=dot((dot(fisher_vec,i)/norm_w),fisher_vec/norm_w)
        co_ordinate=[proj[0]]
        co_ordinate.append(proj[1])
        co_ordinate.append(1)
        l=[array(co_ordinate),1]
        points_1.append(l)
        plt.plot(proj[1],proj[0],"sb")

    for i in class2_data :
        proj=dot((dot(fisher_vec,i)/norm_w),fisher_vec/norm_w)
        co_ordinate=[proj[0]]
        co_ordinate.append(proj[1])
        co_ordinate.append(1)
        l=[array(co_ordinate),0]
        points_1.append(l)
        plt.plot(proj[1],proj[0],"sr")

    ab = fisher_vec[0]/fisher_vec[1]
    plt.plot(space, ab*space, 'm-')
    
    return points_1

def perceptron(data,eta):
    w = array([0.0,0.0,0.0])
    flag=0

    while flag == 0 :
        flag=1
        for i in data :
            x, expected_class = i[0],i[1]
            result = dot(w, x)
            error = expected_class - unit_step(result)
            if error != 0 :
                flag=0
                w=np.add(w,eta * error * x)
    return w

def LMS(data, desired, eta):

    weights = array([0.0,0.0,0.0])
    epsilon = 0.00004
    flag=1
    k=1
    while flag == 1 :
        flag=0
        for x,actual_class in data:
            learning_rate=float(eta)/float(k)
            result=np.dot(x,weights)*actual_class
            e= desired -result 
            weights=np.add(weights,learning_rate * e * x*actual_class)
            check=learning_rate*(desired-np.dot(x*actual_class,weights))
            val=np.multiply(x*actual_class,check)
            if np.linalg.norm(val) > epsilon :
                flag = 1
            k+=1

    return weights

def draw_func(fisher_classifier,lms_classifier,data):
    x = np.arange(-2,4) 
    plt.xlim(-2,4)
    plt.ylim(-2,4)
    l = np.linspace(-2,4)
    plt.xlabel("x axis ") 
    plt.ylabel("y axis ")

    for i,j in data :
        if j== 1 :
            plt.plot(i[0],i[1],"ob")
        else :
            plt.plot(i[0],i[1],"or")

    aa, bb = -fisher_classifier[0]/fisher_classifier[1], -fisher_classifier[2]/fisher_classifier[1]
    plt.plot(l, aa*l+bb, 'g-')
    cc,dd = -lms_classifier[0]/lms_classifier[1],-lms_classifier[2]/lms_classifier[1]
    plt.plot(l, cc*l+dd, 'c-')
    plt.show()



if __name__ == '__main__':
    
    data1=[
    [array([3,3,1]),1],
    [array([3,0,1]),1],
    [array([2,1,1]),1],
    [array([0,2,1]),1],
    [array([-1,1,1]),-1],
    [array([-1,-1,1]),-1],
    [array([0,0,1]),-1],
    [array([1,0,1]),-1],
    ]

    data2=[
    [array([3,3,1]),1],
    [array([3,0,1]),1],
    [array([2,1,1]),1],
    [array([0,1.5,1]),1],
    [array([-1,1,1]),-1],
    [array([0,0,1]),-1],
    [array([-1,-1,1]),-1],
    [array([1,0,1]),-1],
    ]

    projected_points=fisher(data1)
    fisher_classifier = perceptron(projected_points,0.5)

    lms_classifier = LMS(data1, 1,0.5)
    print "LMS classifier :",lms_classifier

    plt.title("Dataset1")
    draw_func(fisher_classifier,lms_classifier,data1)
    
    projected_points=fisher(data2)
    fisher_classifier = perceptron(projected_points,0.5)

    lms_classifier = LMS(data2, 1,0.5)
    print "LMS classifier :",lms_classifier
    
    plt.title("Dataset2")
    draw_func(fisher_classifier,lms_classifier,data2)