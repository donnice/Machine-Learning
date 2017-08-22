from numpy import *

def loadSimpData():
    datMat = matrix([[1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1. , 1.],
        [2. , 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

'''
Stump classify:
Set the minError to +inf
For every feature in the dataset:
    for every step:
        Build a decision stump and test it with the weighted dataset
        If error is less than minError: 
            set this stump as the best stump
return best stump
'''
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq = 'It':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray
