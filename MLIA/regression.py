from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

'''
w = (X.T*X).I*X.T*y
'''
def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0: # get determinant
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

def plotRegres(fileName):
    import matplotlib.pyplot as plt
    xArr, yArr = loadDataSet(fileName)
    ws = standRegres(xArr, yArr)
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat * ws
    print corrcoef(yHat.T, yMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:,1],yHat)
    plt.show()

# Locally weighted linear regression
'''
w = (X.T*W*X).I*X.T*W*y
'''
def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye(m)) # Create diagnoal matrix 
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k = 1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

# test error
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()

def predictAbalone():
    abX, abY = loadDataSet('abalone.txt')

    # Testing error
    yHat01 = lwlrTest(abX[0: 99], abX[0: 99], abY[0: 99], 0.1)
    yHat1 = lwlrTest(abX[0: 99], abX[0: 99], abY[0: 99], 1)
    yHat10 = lwlrTest(abX[0: 99], abX[0: 99], abY[0: 99], 10)
    print rssError(abY[0: 99], yHat01.T)
    print rssError(abY[0: 99], yHat1.T)
    print rssError(abY[0: 99], yHat10.T)

    # Training error: using the smallest kernel will overfit our data
    yHat01 = lwlrTest(abX[100: 199], abX[0: 99], abY[0: 99], 0.1)
    print rssError(abY[100: 199], yHat01.T)
    yHat1 = lwlrTest(abX[100: 199], abX[0: 99], abY[0: 99], 1)
    print rssError(abY[100: 199], yHat1.T)
    yHat10 = lwlrTest(abX[100: 199], abX[0: 99], abY[0: 99], 10)
    print rssError(abY[100: 199], yHat10.T)

    # linear regression error
    ws = standRegres(abX[0: 99], abY[0: 99])
    yHat = mat(abX[100: 199]) * ws
    print rssError(abY[100: 199], yHat.T.A)

# Ridge regression
'''
w = (X.T*X + (\lambda)*I).I*X.T*y
notice that I is identity matrix, while .I means tranverse
'''
def ridgeRegres(xMat, yMat, lam = 0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0) # Compute the variance along the specified axis
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

# Lasso
'''
Pseudo-code:
Regularize the data to have 0 mean and unit variance
For every iteration:
    Set lowestError to inf
    For every feature:
        For increasing and decreasing:
            Change one coefficient to get a new W
            Calculate the error with new W
            If Error is lower than lowestError: 
                set Wbest to current W
        Update set W to Wbest
'''

def plotLWLR():
    xArr, yArr = loadDataSet('ex0.txt')
    lwlr(xArr[0], xArr, yArr, 1.0)
    yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    xMat = mat(xArr)
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:,0,:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0] , s = 2, c = 'red')
    plt.show()

def plotRidgeTest():
    import matplotlib.pyplot as plt
    abX, abY = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()
