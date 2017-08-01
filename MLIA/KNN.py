# Listing 2.1, 2.2
# P 53
from numpy import *
import operator

def createDataSet():
	group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'B', 'C', 'D']
	return group, labels

def classify0(inX, dataSet, labels, k):
	'''
	Distance calculation
	For every point in dataset:
		calculate distance between inX and the current point
		sort distances in increasing order
		take k items with lowest distances to inX
		find the majority class among these items
		return majority class as our prediction for class of inX
	'''
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat**2
	# add each line
	# for example, [[0,1,2],[2,1,3]] => [3,6]
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()
	'''
	Voting with lowest k distances
	'''
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

def file2matrix(filename):
	fr = open(filename)
	numberOfLines = len(fr.readlines()) # Get # of lines
	returnMat = zeros((numberOfLines, 3)) # Create Numpy matrix to return
	classLabelVector = []
	fr = open(filename)
	# Parse line to a list
	index = 0
	for line in fr.readlines():
		line = line.strip() # remove all whitespaces at start and end
		listFromLine = line.split('\t') # \t is a tab or 6 spaces
		returnMat[index, :] = listFromLine[0:3]
		# print returnMat
		classLabelVector.append(listFromLine[-1])
		index += 1
	return returnMat, classLabelVector

def autoNorm(dataSet):
	'''
	>> tile(5,(3,2))
	array([[5, 5],
       [5, 5],
       [5, 5]])
	'''
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals, (m, 1))
	normDataSet = normDataSet/tile(ranges, (m, 1))	
	return normDataSet, ranges, minVals

def datingClassTest():
	hoRatio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], \
					datingLabels[numTestVecs: m], 3)
		print "the classifier came back with: %s, the real answer is: %s" \
					% (classifierResult, datingLabels[i])
		if (classifierResult != datingLabels[i]):
			errorCount += 1.0
	print "The total error rate is: %f" % (errorCount/float(numTestVecs))