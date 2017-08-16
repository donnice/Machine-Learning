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
# Helper function for SMO (Sequential Minimal Optimization) algorithm
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineAr[2]))
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
