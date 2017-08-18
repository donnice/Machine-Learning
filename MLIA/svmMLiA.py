'''
SVM: The best margin (seperating line in 2-D plan) of points
Suport vectors: Points closet to hyperplane
Point perpendicular to line: |w(T)x+b|/||w||
label: 1 or -1, the nearest point(s)
margin maximum with points with smallest margin:
    argmax(w,b){min(label*(w(T)x+b))/||w||}
=> Lagrange =>
    max(a)[sigma(1, m){a} - sigma(i,j=1,, m){label(i)*label(j)*ai*aj*<x(i),x(j)}/2]
    with a>=0 and sigma(i-1, m){ai*label(i)} = 0
'''

from numpy import *

# Helper function for SMO (Sequential Minimal Optimization) algorithm
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

'''
Pseudo code for simplified SMO:

Create an alphas vector filled with 0s
While the number of iterations is less than MaxIterations:
    For every data vector in the dataset:
        If the data vector can be optimized:
            Select another data vector at random
            Optimize the two vectors together
            If the vector cannot be optimized -> break
    If no vector were optimized -> increse the iteration count
'''
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iterance = 0
    while(iterance < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T*\
                       (dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            # Enter optmiz if alphas can be changed
            if((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or \
                ((labelMat[i]*Ei > toler) and \
                (alphas[i] > 0)):
                # randomly select second alpha
                j = selectJrand(i, m)

                fXj = float(multiply(alphas, labelMat).T*\
                           (dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # Guarentee alphas stay between 0 and C
                if(labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, C + alphas[i] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print "L == H"
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0: #best optimized value of alpha[j]
                    print "eta>=0"
                    continue

                # Update i by same amount as j in opposite direction
                alphas[j] -= labelMat[j] * (Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if(abs(alphas[j] - alphaJold) < 0.00001):
                    print "j not moving enough"
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * \
                          (alphaJold - alphas[j])
                # Set the constant terms
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                     dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j]*(alphas[j] - alphaJold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j]*(alphas[j] - alphaJold) * \
                     dataMatrix[j, :] * dataMatrix[j, :].T
                if(0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif(0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % \
                                  (iterance, i, alphaPairsChanged)
        if(alphaPairsChanged == 0):
            iterance += 1
        else:
            iterance = 0
        print "iteration number: %d" % iterance
    return b, alphas


                