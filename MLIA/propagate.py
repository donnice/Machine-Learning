import os
import numpy as np
 
# An example in that book, the training set and parameters' sizes are fixed
training_set = np.array([[[3, 3], 1], [[4, 3], 1], [[1, 1], -1]])
 
a = np.array([0, 0, 0])
b = 0
Gram = np.array([])
y = np.array(range(len(training_set))).reshape(1, 3)# used for labels
x = np.array(range(len(training_set)) * 2).reshape(3, 2)# used for features
 
# calculate the Gram matrix
def cal_gram():
    g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for i in range(len(training_set)):
        for j in range(len(training_set)):
            g[i][j] = np.dot(training_set[i][0], training_set[j][0])
    return g
 
# update parameters using stochastic gradient descent
def update(i):
    global a, b
    a[i] = a[i] + 1
    b = b + training_set[i][1]
    print a, b # you can uncomment this line to check the process of stochastic gradient descent
 
# calculate the judge condition
def cal(key):
    global a, b, x, y
    i = 0
    for item in training_set:
        y[0][i] = item[1]
        i = i + 1
    temp = a * y
    res = np.dot(temp, Gram[key])
    res = (res + b) * training_set[key][1]
    return res[0]
 
# check if the hyperplane can classify the examples correctly
def check():
    global a, b, x, y
    flag = False
    for i in range(len(training_set)):
        if cal(i) <= 0:
            flag = True
            update(i)
    if not flag:
        i = 0
        for item in training_set:
            y[0][i] = item[1]
            x[i] = item[0]
            i = i + 1
        temp = a * y
        w = np.dot(temp, x)
        print "RESULT: w: " + str(w) + " b: "+ str(b)
        os._exit(0)
    flag = False
     
 
if __name__=="__main__":
    Gram = cal_gram()# initialize the Gram matrix
    for i in range(1000):
        check()
    print "The training_set is not linear separable. "