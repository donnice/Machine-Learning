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
# Compare data by threshold
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq = 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros(m, 1))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = \
                    stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
    return bestStump, minError, bestClasEst

'''
AdaBoost training with decision stumps:
For each iteration:
    Find the best stump using buildStump()
    Add the best stump to the stump array
    Calculate alpha
    Calculate new weight vector -D
    Update the aggregate class estimate
    If the error rate == 0.0:
        break out of the for loop
'''

