from sklearn.datasets import load_iris

from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import load_iris

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from scipy.spatial import distance

from sklearn.metrics import accuracy_score
	

def KNN():

	data = pd.read_csv("PlayPredictor.csv")

	Wether = data.Wether

	Temperature = data.Temperature

	Play = data.Play


	le = LabelEncoder()

	WetherX = le.fit_transform(Wether)

	TemperatureX = le.fit_transform(Temperature)
	
	PlayX = le.fit_transform(Play)

	DataX = zip(WetherX , TemperatureX)

	dobj = KNeighborClassifier(n_neighbors = 3)

	dobj.fit(list(DataX),PlayX)

	result = dobj.predict([[0,2]])

	if result==1:

		print("You can play")

	else:
		
		print("You cant play")	




def main():

	KNN()	


if __name__ == '__main__':
	main()
