import pandas as pd
import numpy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt

def readData(filename):
    data=[]
    with open(filename, "r") as f:
        for line in f.readlines():
            words = line.split(",")
            words[0] = float(words[0])
            words[1] = float(words[1])
            words[2] = float(words[2])
            words[3] = float(words[3])
            words[4] = words[4].replace("\n","")
            data.append(words)
    return data

def iris():
	iris_data = readData("iris.data")
	iris_x=[]  #feature
	iris_y=[]  #target
	for i in range(len(iris_data)):
		iris_x.append(iris_data[i][:4])
		iris_y.append(iris_data[i][4])
	'''
	pca=PCA(n_components='mle')
	new_iris_x=pca.fit_transform(iris_x)
	explained_variance = numpy.var(new_iris_x, axis=0)
	explained_variance_ratio = explained_variance / numpy.sum(explained_variance)
	print(explained_variance)
	print(explained_variance_ratio)
	'''
	train_x, test_x, train_y, test_y = train_test_split(iris_x, iris_y, test_size=0.3)

	#Decision Tree
	decision_tree= tree.DecisionTreeClassifier()
	iris_decision_tree = decision_tree.fit(train_x, train_y)
	DT_test_y_predicted = iris_decision_tree.predict(test_x)
	DT_accuracy = metrics.accuracy_score(test_y, DT_test_y_predicted)

	#KNN
	knn = neighbors.KNeighborsClassifier(n_neighbors = 5)
	iris_knn = knn.fit(train_x, train_y)
	KNN_test_y_predicted = iris_knn.predict(test_x)
	KNN_accuracy = metrics.accuracy_score(test_y, KNN_test_y_predicted)

	#Naive Bayes
	nb = GaussianNB()
	iris_nb = nb.fit(train_x, train_y)
	NB_test_y_predicted = iris_nb.predict(test_x)
	NB_accuracy = metrics.accuracy_score(test_y, NB_test_y_predicted, normalize=True)

	#Accuarcy
	print("Iris---------------------------------------")
	print("DecsionTree = " + str(round(DT_accuracy, 7)))
	print("KNN         = " + str(round(KNN_accuracy,7)))
	print("NaiveBayes  = " + str(round(NB_accuracy, 7)))
	print("")

def fire():
	fire_data = pd.read_csv("forestfires.csv")
	#print(fire_data.dtypes)
	obj_fire_data = fire_data.select_dtypes(include=['object']).copy()
	#print(obj_fire_data)
	#pd.options.display.max_columns=20
	#pd.options.display.max_rows=20
	dummies_data = pd.get_dummies(obj_fire_data, columns=["month","day"])
	#print(pd.get_dummies(obj_fire_data, columns=["month","day"]))
	new_data = pd.concat([fire_data.select_dtypes(exclude=['object']), dummies_data], axis=1)

	#standardlize
	#for key in new_data.loc[:, new_data.columns != 'area']:
	#	new_data[key]=preprocessing.scale(new_data[key].astype('float64'))

	#normalize
	feature_data = new_data.loc[:, new_data.columns != 'area']
	nor_data = preprocessing.normalize(feature_data.astype('float64'), norm='l2')

	# redefine area class
	for row in new_data.area:
		if row < 0.001:
			new_data.area.replace({row: 0}, inplace=True)
		elif row >= 0.001 and row < 1:
			new_data.area.replace({row: 1}, inplace=True)
		elif row >= 1 and row < 10:
			new_data.area.replace({row: 2}, inplace=True)
		elif row >= 10 and row < 100:
			new_data.area.replace({row: 3}, inplace=True)
		elif row >= 100 and row < 1000:
			new_data.area.replace({row: 4}, inplace=True)
		elif row >= 1000:
			new_data.area.replace({row: 5}, inplace=True)

	fire_x = nor_data
	fire_y = new_data.area.astype('int64')
	#PCA
	'''
	pca = PCA(n_components=3)
	new_fire_x = pca.fit_transform(fire_x)
	explained_variance = numpy.var(new_fire_x, axis=0)
	explained_variance_ratio = explained_variance / numpy.sum(explained_variance)
	print(explained_variance)
	print(explained_variance_ratio)
	'''
	train_x, test_x, train_y, test_y = train_test_split(fire_x, fire_y, test_size=0.3)

	#Decision Tree
	decision_tree= tree.DecisionTreeClassifier()
	fire_decision_tree = decision_tree.fit(train_x, train_y)
	DT_test_y_predicted = fire_decision_tree.predict(test_x)
	DT_accuracy = metrics.accuracy_score(test_y, DT_test_y_predicted)

	#KNN
	knn = neighbors.KNeighborsClassifier(n_neighbors = 10)
	fire_knn = knn.fit(train_x, train_y)
	KNN_test_y_predicted = fire_knn.predict(test_x)
	KNN_accuracy = metrics.accuracy_score(test_y, KNN_test_y_predicted)

	#Naive Bayes
	nb = BernoulliNB(alpha=1.0)
	fire_nb = nb.fit(train_x, train_y)
	NB_test_y_predicted = fire_nb.predict(test_x)
	BNB_accuracy = metrics.accuracy_score(test_y, NB_test_y_predicted, normalize=True)
	nb = MultinomialNB(alpha=1.0)
	fire_nb = nb.fit(train_x, train_y)
	NB_test_y_predicted = fire_nb.predict(test_x)
	MNB_accuracy = metrics.accuracy_score(test_y, NB_test_y_predicted, normalize=True)

	#Accuarcy
	print("ForestFire---------------------------------")
	print("DecsionTree = " + str(round(DT_accuracy, 7)))
	print("KNN         = " + str(round(KNN_accuracy,7)))
	print("BNaiveBayes = " + str(round(BNB_accuracy,7)))
	print("MNaiveBayes = " + str(round(MNB_accuracy,7)))

if __name__ == "__main__":
	iris()
	fire()