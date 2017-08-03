# Pesudo code
'''
Check if every item in the dataset is in the same class: 
    If so return the class label
    Else
        find the best feature to split the data 
        split the dataset
        create a branch node
            for each split
                call createBranch and add the result to the branch node
        return branch node
'''

from math import log

# H = -Sum(1,n)(p(xi)log(p(xi))), were p(xi) is the prob of choosing this class
def calShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(labelCounts, 0) + 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt
