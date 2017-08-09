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
import operator
from math import log

# H = -Sum(1,n)(p(xi)log(p(xi))), were p(xi) is the prob of choosing this class
def calcShannonEnt(dataSet):
    '''
    Disorder of the dataSet
    '''
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        # labelCounts[currentLabel] = labelCounts.get(labelCounts, 0) + 1
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'Yes'],
            [1, 1, 'Yes'],
            [1, 0, 'No'],
            [0, 1, 'No'],
            [0, 1, 'No']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# extend: way to append multiple elements in the list
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # cut out the feature split on
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# Use entropy to tell which split best organizes the data
def chooseBestFeatureToSplit(dataSet):
    '''
    The information gain is the reduction in entropy
    '''
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # Create unique list of class labels
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        # Calculate entropy for each split
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        # find the best infomation gain
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),
    key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # Stop when all classes are equal
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # When no more features, return majority
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    # get list of unique values
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet\
        (dataSet, bestFeat, value), subLabels)
    return myTree

# Classification function for an existing decision tree
# Doesn't understand
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr) # translate label string to index
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

# Persisting the decision tree
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
