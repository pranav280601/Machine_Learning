from sklearn.datasets import load_iris

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from scipy.spatial import distance

from sklearn.metrics import accuracy_score



def CalculateDistance(X,Y):
	return distance.euclidean(X,Y)


class Pranav:


	def fit(self,TrainningData,TrainningTarget):

		self.TrainningData = TrainningData
		self.TrainningTarget = TrainningTarget
		print("Data Trained successfully")

	def ShortestDistance(self , row):

		MinIndex = 0
		MinDistance = CalculateDistance(row , self.TrainningData[0])

		for i in range(1,len(self.TrainningData)):

			Distance = CalculateDistance(row,self.TrainningData[i])

			if Distance < MinDistance:
				MinDistance = Distance
				MinIndex=i

		return self.TrainningTarget[MinIndex] 


	def predict(self,TestingData):

		predicitions = []

		for row in TestingData:
			label=self.ShortestDistance(row)
			predicitions.append(label)
		return predicitions

def KNN():
	
	Line = "*"*50

	iris = load_iris()

	print("Iris Dataset Loaded successfully")


	data = iris.data
	target = iris.target

	print(Line)
	print("Actual Dataset")
	print(Line)
	for i in range(len(iris.target)):
		print("ID : %d Label : %s, Feature : %s" %(i,iris.data[i], iris.target[i]))

	data_train , data_test ,target_train , target_test = train_test_split(data , target , test_size = 0.5)

	print(Line)
	print("Training Data set")
	print(Line)
	for i in range(len(data_train)):
		print("ID : %d Data : %s, Feature : %s" %(i,data_train[i], target_train[i]))
            
	print(Line)
	print("Testing Data set")
	print(Line)
	for i in range(len(data_test)):
		print("ID : %d Data : %s, Feature : %s" %(i,data_test[i], target_test[i]))
    
	print(Line)

	vobj = Pranav()

	vobj.fit(data_train,target_train)

	ret=vobj.predict(data_test)


	icnt = 0
	for i in range(len(data_test)):
		if target_test[i] != ret[i]:
			icnt = icnt + 1
	print("Number of wrong answers by the ML model : ",icnt)
	print(Line)
    
	Accuracy = accuracy_score(target_test , ret)


	return Accuracy

def main():


	ret= KNN()

	print("Accuracy of KNN is:",ret*100,"%")

if __name__ == '__main__':
	main()