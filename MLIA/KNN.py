# Listing 2.1, 2.2
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
			

