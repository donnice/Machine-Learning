# General approach to logistic regression
'''
1. Collect
2. Prepare: Numeric values are needed for distance calculation
3. Analyze
4. Train: Try to find optimal coefficients to classify the data
5. Test
6. Use - regression calculation

sigmoid function:
σ(z) = 1/(1+e(-z))
z = w0x0 + w1x1 + ... + wnxn

Gradient Ascent algorithm:
w: = w + α(Gradient)wf(w)
'''

# Logistic regression gradient ascent optimization functions
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat