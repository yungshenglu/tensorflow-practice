import numpy as np
import math
import random
from tree import Node
from material import *

def RF():
	npdtype = [('s_len','f16'),('s_wid','f16'),('p_len','f16'),('p_wid','f16'),('class','U32')]
	data = readData()
	data = np.array(data, dtype=npdtype)
	np.random.shuffle(data)
	
	accu = 0.0
	pre_class0 = 0.0
	rec_class0 = 0.0
	pre_class1 = 0.0
	rec_class1 = 0.0
	pre_class2 = 0.0
	rec_class2 = 0.0
	# K fold with K=5 
	K = 5
	splitList = np.split(data,K)
	for i in range(K):
		trainingData = np.array([],dtype=npdtype)
		testingData = splitList[i]
		for j in range(K):
			if not i == j:
				trainingData = np.concatenate((trainingData,splitList[j]),axis=0)

		tree_num = 10
		root=None  #RF's root is defined in validate()
		acc_n, pre_n, rec_n = validate(root,testingData,trainingData,tree_num)
		accu += acc_n
		pre_class0 += pre_n['Iris-setosa']
		rec_class0 += rec_n['Iris-setosa']
		pre_class1 += pre_n['Iris-versicolor']
		rec_class1 += rec_n['Iris-versicolor']
		pre_class2 += pre_n['Iris-virginica']
		rec_class2 += rec_n['Iris-virginica']

	print(round(accu/K, 7))
	print(round(pre_class0/K, 7), round(rec_class0/K, 7))
	print(round(pre_class1/K, 7), round(rec_class1/K, 7))
	print(round(pre_class2/K, 7), round(rec_class2/K, 7))



if __name__ == "__main__":
    RF()