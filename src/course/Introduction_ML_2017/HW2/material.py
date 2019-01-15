import csv
import math
from operator import itemgetter

class Node():
    def __init__(self,data):
        self.ID = -1
        self.data = data
        self.right = None
        self.left = None
        self.threshold = -1
        self.feature_index = 2
        self.leaf = False
        self.parent = None

    def is_leaf(self,leaf):
        self.leaf = True

    def set_ID(self,ID):
        self.ID = ID

    def get_parent(self, parent):
        self.parent = parent

    def set_threshold(self,threshold):
        self.threshold = threshold

    def set_feature_index(self,index):
        self.feature_index = index


def readCSV(filename):
    data = []
    with open(filename) as csvfile:
        flag=0   #ignore first row in file
        c=0
        content = csv.reader(csvfile,delimiter=',')
        for row in content:
            if len(row) > 0 and flag!=0:
                c+=1
                for i in range(2,11):
                    row[i] = float(row[i])
                data.append(row)
            flag = 1
    return data

def makeTree(currnode, attrNum):         #K-d tree, split by attrNum
    if(len(currnode.data)==1):
        currnode.set_ID(currnode.data[0][0])   #get instance's index
        currnode.is_leaf(True)
        currnode.set_feature_index(attrNum)    #useless
        return currnode
    elif(len(currnode.data)>0):
        sorted_data = sorted(currnode.data, key=itemgetter(attrNum))
        m = math.ceil(len(currnode.data)/2)-1
        median_value = sorted_data[m][attrNum]
        while m >= 1 and sorted_data[m-1][attrNum] == sorted_data[m][attrNum]:
            m -= 1
        currnode.set_ID(sorted_data[m][0])     #get instance's index
        currnode.set_threshold(median_value)   #get median value
        currnode.set_feature_index(attrNum)    #get attrNum 2~10
        attrNum = (((attrNum-2) + 1) % 9) + 2       # 9 attributes  2~10
        if m!=0:
            currnode.left = Node(sorted_data[:m])
            currnode.left.get_parent(currnode)
            makeTree(currnode.left, attrNum)   
        else:    
            # the except ocurr when the node only have right child
            # To avoid getting lost when searching, that the node be its left child
            currnode.left = Node(sorted_data[0])
            currnode.left.get_parent(currnode)
            currnode.left.is_leaf(True) 
        currnode.right = Node(sorted_data[m+1:])
        currnode.right.get_parent(currnode)
        makeTree(currnode.right, attrNum)

def distance(a, b):
    dis_square = 0
    for i in range(2,11):
        dis_square += (a[i]-b[i])*(a[i]-b[i])
    return math.sqrt(dis_square)

def searchLeaf(instance,node, visit1):
    while node is not None and node.ID not in visit1:
        visit1.append(node.ID)
        if node.leaf==True:
            break
        if instance[node.feature_index] < node.threshold:
            node = node.left
        else:
            node = node.right
    return node

def voteMajority(neighbors, trainingData):
    count = {}
    for instance in neighbors:
        if not trainingData[int(instance[0])][11] in count:
            count[trainingData[int(instance[0])][11]] = 1
        else:
            count[trainingData[int(instance[0])][11]] += 1
    majority = max(count, key=lambda i: count[i])
    return majority


    