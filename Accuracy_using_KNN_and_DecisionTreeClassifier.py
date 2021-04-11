from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree 
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def Decision():

	dataset = load_iris()

	data = dataset.data
	target = dataset.target

	data_train,data_test,target_train,target_test = train_test_split(data,target,test_size = 0.5)


	dobj = tree.DecisionTreeClassifier()

	dobj.fit(data_train,target_train)

	result = dobj.predict(data_test)

	Accuracy = accuracy_score(target_test,result)

	print("Accuracy of DecisionTreeClassifier is:",(Accuracy*100),'%')



def KNN():				# knn - k nearest neighbour

	dataset = load_iris()

	data = dataset.data
	target = dataset.target

	data_train,data_test,target_train,target_test = train_test_split(data,target,test_size = 0.5)


	dobj = KNeighborsClassifier()

	dobj.fit(data_train,target_train)

	result = dobj.predict(data_test)

	Accuracy = accuracy_score(target_test,result)

	print("Accuracy using KNN algo is:",(Accuracy*100),'%')



def main():

	Decision()
	KNN()


if __name__ == '__main__':
	main()