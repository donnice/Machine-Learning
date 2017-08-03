'''
1. Collect: Any method.
2. Prepare: Numeric values are needed for a distance calculation. A structured data format is best.
3. Analyze: Any method.
4. Train: Does not apply to the kNN algorithm.
5. Test: Calculate the error rate.
6. Use: This application needs to get some input data and output structured num- eric values. 
Next, the application runs the kNN algorithm on this input data and determines which class the input data should belong to. 
The application then takes some action on the calculated class.
'''

from numpy import *
import operator
import os

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

def classifyPerson():
	resultList = ['not at all', 'in small doses', 'in large doses']
	percentTats = float(raw_input(\
				"percentage of time spent plaing video games?"))
	ffMiles = float(raw_input("frequent flier miles earned per year?"))
	iceCream = float(raw_input("liters of ice cream consumed per year?"))
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
	print "You will probably like this person: "+resultList[int(classifierResult) - 1]

def img2vector(filename):
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(lineStr[j])
	return returnVect

def handwritingClassTest():
	hwLabels = []
	trainingFileList = os.listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m, 1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNameStr = int(fileStr.split('_')[0])
		hwLabels.append(classNameStr)
		trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
	testFileList = os.listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNameStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		if (classifierResult != classNameStr):
			errorCount += 1.0
		print "\nthe total number of error is: %d" % errorCount
		print "\nthe total error rate is: %f" % (errorCount/float(mTest))
