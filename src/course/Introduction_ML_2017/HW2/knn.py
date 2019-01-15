from material import *
import sys

def KNN(k):
	train_data = readCSV(sys.argv[1])
	test_data = readCSV(sys.argv[2])
	root = Node(train_data)
	makeTree(root,2)  #attrNum start from 2
	correct = 0
	instance_count = 0
	output = []
	for q in test_data:
		visit1 = []    #fist time search path
		visit2 = []    #back tracking path
		'''
		nearest_node = root
		nearest_dis = float('inf')
		neighbors = [['-1', float('inf')]]
		'''
		x = searchLeaf(q,root, visit1)

		while x is not None:
			'''
			if distance(q, train_data[int(x.ID)]) < nearest_dis:  # dist(query, data.index=back_parent_ID) < min
				nearest_node = x
				nearest_dis = distance(q,train_data[int(x.ID)])
			'''
			neighbors = sorted(neighbors, key=itemgetter(1))
			if distance(q, train_data[int(x.ID)]) < neighbors[len(neighbors)-1][1] and x.ID not in visit2:
				if len(neighbors) == k:
					neighbors.pop()
				neighbors.append([x.ID, distance(q,train_data[int(x.ID)])])

			if x.ID != root.ID and x.ID != root.left.ID and x.ID != root.right.ID:  #not root rootleft rootright
				back = x.parent
				#print(back.ID)
				while (back.parent is not None and back.ID in visit2):
					back = back.parent
				if back.ID == root.ID and root.ID in visit2:   # backtrack done, avoid to going through root twice
					break
				visit2.append(back.ID)
			else:
				break

			'''
			if abs(q[back.feature_index]-back.threshold) < nearest_dis:
				if distance(q, train_data[int(back.ID)]) < nearest_dis:
					nearest_node = back
					nearest_dis = distance(q,train_data[int(back.ID)])
			'''
			neighbors = sorted(neighbors, key=itemgetter(1))
			if abs(q[back.feature_index]-back.threshold) < neighbors[len(neighbors)-1][1]:
				if distance(q, train_data[int(back.ID)]) < neighbors[len(neighbors)-1][1]:
					if len(neighbors) == k:
						neighbors.pop()
					neighbors.append([back.ID, distance(q,train_data[int(back.ID)])])
				if len(back.right.data)>0 and len(back.left.data)>0:
					if q[back.feature_index] < back.threshold:
						x = searchLeaf(q, back.right, visit1)
					else:
						x = searchLeaf(q, back.left, visit1)
				else:
					x = back
			else:
				x = back
				
		#validate
		#guess_label = train_data[int(nearest_node.ID)][11]
		guess_label = voteMajority(neighbors, train_data)
		if q[11] == guess_label:
			correct += 1

		if instance_count < 3:
			neighbors = sorted(neighbors, key=itemgetter(1))
			output.append(neighbors)
		instance_count += 1

	print("KNN accuracy: " + str(round(correct/len(test_data), 7)))
	for i in range(3):
		for j in range(k):
			print(output[i][j][0], end=" ")
		print("")
	print("")


if __name__ == "__main__":
    KNN(1)
    KNN(5)
    KNN(10)
    KNN(100)