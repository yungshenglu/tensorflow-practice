import numpy as np

class Node():
    def __init__(self,data):
        self.data = data
        self.right = None
        self.left = None
        self.threshold = -1
        self.feature_index = -1
        self.label = ""
        self.leaf = False

    def set_threshold(self,threshold):
        self.threshold = threshold

    def set_feature_index(self,index):
        self.feature_index = index

    def set_label(self,label):
        self.label = label