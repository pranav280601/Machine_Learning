import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from matplotlib.pyplot import figure,show
from seaborn import countplot 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



def TitanicLogistic(): 

	# Step 1 - --------------------  Load Data  -------------------------------------


	Titanic_data = pd.read_csv("TitanicDataset.csv")


	# Data Rangling  -  Data Analysis and Data Cleaning 


	# Step 2 - -------------------------- Data Analysis -------------------------


	print("Visualtion of Survived and Non-Survived passengers")

	figure()

	countplot(data = Titanic_data,x = "Survived").set_title("Survived V/s Non-Survived")

	#show()

	print("Visualisation according to Gender")

	figure()

	countplot(data = Titanic_data,x = "Survived",hue = "Sex").set_title("Visualisation according to Sex")

	#show()

	print("Visualisation based on Passenger Class")

	figure()

	countplot(data = Titanic_data,x = "Survived",hue = "Pclass").set_title("Survived V/s Non-Survived based on Pclass")

	#show()

	print("Visualisation according to age")

	figure()

	#countplot(data = Titanic_data,x = "Survived",hue = "Age").set_title("Visualisation according to age")

	Titanic_data["Age"].plot.hist().set_title("Visualisation according to age")

	#show()



	# Step 3 -  ---------------------------------  Data Cleaning  -----------------------------------------


	Titanic_data.drop("zero", axis= 1, inplace = True)

	print("Data after coloumn removal")

	#print(Titanic_data.head())

	Sex = pd.get_dummies(Titanic_data["Sex"])

	#print(Sex.head())

	Sex = pd.get_dummies(Titanic_data["Sex"],drop_first = True)

	print("Sex coloumn after updation")

	#print(Sex.head())

	Pclass = pd.get_dummies(Titanic_data["Pclass"])

	Pclass = pd.get_dummies(Titanic_data["Pclass"],drop_first = True)

	#print(Pclass.head())
	
	# Concate  Sex  and  Pclass field  in our dataset

	Titanic_data = pd.concat( [Titanic_data,Sex,Pclass] , axis = 1 )

	print("DAta after concatination")

	#print(Titanic_data.head())	

	print("Removing unessesary data")

	Titanic_data.drop( ["Sex","Embarked","Pclass","Parch","sibsp"], axis = 1, inplace = True)

	#Titanic_data.drop("Sex", axis= 1, inplace = True)

	#Titanic_data.drop("Embarked", axis= 1, inplace = True)

	#Titanic_data.drop("Pclass", axis= 1, inplace = True)

	#Titanic_data.drop("Parch", axis= 1, inplace = True)

	print(Titanic_data.head())

	# --------------------   Divide data set into x and y ----------------------

	X = Titanic_data.drop("Survived", axis = 1)

	Y = Titanic_data["Survived"]


	# -------------- Split the data for training and testing purpose ----------

	X_train,X_test,Y_train,Y_test = train_test_split(X , Y , test_size = 0.5)


	obj = LogisticRegression(max_iter = 1000)


	# Step - 4 --------------  Training the dataset ---------------------

	obj.fit(X_train,Y_train)


	# Step - 5 --------------- Testing the dataset ------------------

	ret = obj.predict(X_test)


	print("Accuracy of the given dataset is:",(accuracy_score(ret,Y_test)*100))

	print("Confusion matrix is")

	print(confusion_matrix(Y_test,ret))






def main():

	print("Logistic Case Study")

	

	TitanicLogistic()




if __name__ == '__main__':
	main()