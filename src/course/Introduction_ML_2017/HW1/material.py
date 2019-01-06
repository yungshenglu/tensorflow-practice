import numpy as np
import math
import random
from tree import Node
def readData():
	data=[]
	with open("iris.data", "r") as f:
		for line in f.readlines():
			words = line.split(",")
			words[0] = float(words[0])
			words[1] = float(words[1])
			words[2] = float(words[2])
			words[3] = float(words[3])
			words[4] = words[4].replace("\n","")
			data.append(tuple(words))
	return data

#########Construct Desicion Tree #########
def calculateEntropy(data):
	count = {}
	entropy = 0.0
	for instance in data:
		if not instance[4] in count:
			count[instance[4]] = 1
		else:
			count[instance[4]] += 1
	for freq in count.values():
		entropy += -(freq/len(data)) * math.log(freq/len(data), 2) 
	return entropy

def minRemAndThreshold(data,attrIndex):
	min_rem = float('inf')
	rem = float('inf')
	best_threshold = float('inf')
	split = 0
	sortedData = sorted(data,key=lambda x: x[attrIndex])
	for i in range(len(sortedData)-1):
		if sortedData[i][4] != sortedData[i+1][4]:
			leftData = []
			rightData = []
			threshold = (sortedData[i][attrIndex] + sortedData[i+1][attrIndex])/2
			# split instances by threshold
			for instance in sortedData:
				if instance[attrIndex] < threshold:
					leftData.append(instance)
				else:
					rightData.append(instance)
			rem = calculateEntropy(leftData)*(len(leftData)/len(sortedData)) + calculateEntropy(rightData)*(len(rightData)/len(sortedData))
			if rem < min_rem:
				min_rem = rem
				best_threshold = threshold

	return min_rem, best_threshold

def chooseBestFeature(data, RF):
	best_rem = float('inf')
	best_threshold = float('inf')
	best_feature = 0
	if RF:
		sample = random.sample(range(4), 2)
		for attrIndex in sample:
			rem, threshold = minRemAndThreshold(data,attrIndex)
			if rem < best_rem:
				best_rem = rem
				best_threshold = threshold
				best_feature = attrIndex

	else:
		for attrIndex in range(4):
			rem, threshold = minRemAndThreshold(data,attrIndex)
			if rem < best_rem:
				best_rem = rem
				best_threshold = threshold
				best_feature = attrIndex
	return best_threshold, best_feature

def voteMajority(data):
	count = {}
	for instance in data:
		if not instance[4] in count:
			count[instance[4]] = 1
		else:
			count[instance[4]] += 1
	majority = max(count, key=lambda i: count[i])
	return majority

def isPure(node):
	sameclasscount = 0
	target = node.data[0][4]    #take first instance's class as compare standard
	for instance in node.data:
		if target == instance[4]:
			sameclasscount += 1
	if sameclasscount == len(node.data):
		return True
	else:
		return False

def makeTree(currnode, RF=None):
	leftData = []
	rightData = []
	if RF:
		threshold, feature_idx = chooseBestFeature(currnode.data, RF=True)
	else:
		threshold, feature_idx = chooseBestFeature(currnode.data, RF=False)
	currnode.set_threshold(threshold)
	currnode.set_feature_index(feature_idx)

	for instance in currnode.data:
		if instance[feature_idx] < threshold:
			leftData.append(instance)
		else:
			rightData.append(instance)

	if isPure(currnode):    # all instances are the same class
		currnode.set_label(currnode.data[0][4])
		return currnode
	elif len(leftData)==0 or len(rightData)==0:   # threshold can't split data into two parts
		majority = voteMajority(currnode.data)
		currnode.set_label(majority)
		return currnode
	else:
		currnode.left = Node(leftData)
		makeTree(currnode.left)
		currnode.right = Node(rightData)
		makeTree(currnode.right)

#########Calculate Accuracy#########
def classify(node,testingInstance):  
	if len(node.label)>0:   #Has label only if the node is leaf
		return node.label
	if testingInstance[node.feature_index] < node.threshold:
		return classify(node.left, testingInstance)
	else:
		return classify(node.right, testingInstance)

def validate(root, testingData, trainingData, treeNum):
	correct = 0
	data_count={
		'Iris-setosa':0,
		'Iris-versicolor':0,
		'Iris-virginica':0
	}
	TP_count={
		'Iris-setosa':0,
		'Iris-versicolor':0,
		'Iris-virginica':0
	}
	predict_count={
		'Iris-setosa':0,
		'Iris-versicolor':0,
		'Iris-virginica':0
	}
	precision={
		'Iris-setosa':0.0,
		'Iris-versicolor':0.0,
		'Iris-virginica':0.0
	}
	recall={
		'Iris-setosa':0.0,
		'Iris-versicolor':0.0,
		'Iris-virginica':0.0
	}

	for instance in testingData:
		data_count[instance[4]] += 1
		if treeNum > 1:    #Forest
			majority={}
			for n in range(treeNum):
				sample_trainingData = [trainingData[x] for x in random.sample(range(len(trainingData)), 40)]   #bagging
				root = Node(sample_trainingData)
				makeTree(root)
				predict_class = classify(root,instance)
				if not predict_class in majority:
					majority[predict_class] = 1
				else:
					majority[predict_class] += 1
			predict = max(majority, key=lambda p: majority[p])
			predict_count[predict] += 1
			correct = totalcorrect(predict, correct, instance)
			if predict == instance[4]:
				TP_count[instance[4]] += 1
		else:
			predict = classify(root,instance)
			predict_count[predict] += 1
			correct = totalcorrect(predict, correct, instance)
			if predict == instance[4]:
				TP_count[instance[4]] += 1

	#accuarcy
	accuracy = correct / len(testingData)
	#precision
	precision['Iris-setosa'] =TP_count['Iris-setosa'] / predict_count['Iris-setosa']
	precision['Iris-versicolor'] =TP_count['Iris-versicolor'] / predict_count['Iris-versicolor']
	precision['Iris-virginica'] =TP_count['Iris-virginica'] / predict_count['Iris-virginica']
	#recall
	recall['Iris-setosa'] =TP_count['Iris-setosa'] / data_count['Iris-setosa']
	recall['Iris-versicolor'] =TP_count['Iris-versicolor'] / data_count['Iris-versicolor']
	recall['Iris-virginica'] =TP_count['Iris-virginica'] / data_count['Iris-virginica']
	return (accuracy, precision, recall)


def totalcorrect(predict, correct, instance):
	if predict == instance[4]:
		correct += 1
	return correct