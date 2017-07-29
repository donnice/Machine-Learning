# Listing 2.1
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
