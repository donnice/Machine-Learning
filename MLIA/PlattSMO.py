from numpy import *
from svmMLiA import *

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.aplhas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2))) # Error cache

    def calcEk(oS, k):
        fXk = float(multiply(oS.alphas, oS.labelMat).T * \
              (oS.X * oS.X[k, :].T)) + oS.b
        Ek = fXk - float(oS.labelMat[k])
        return Ek

    # Inner-loop heuristic
    def selectJ(i, oS, Ei):
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        oS.eCache[i] = [1, Ei]
        validEcacheList = nonzero(oS.eCache[:, 0].A)[0] # asarray
        if(len(validEcacheList)) > 1:
            for k in validEcacheList:
                if k == i: continue
                Ek = calcEk(oS, k)
                deltaE = abs(Ei - Ek)

                # choose j for maximum step size
                if (deltaE > maxDeltaE):
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:
            j = selectJrand(i, oS.m)
            Ej = calcEk(oS, j)
        return j, Ej

    def updateEk(oS, k):
        Ek = calcEk(oS, k)
        oS.eCache[k] = [1, Ek]