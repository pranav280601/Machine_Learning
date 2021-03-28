

from sklearn import tree 				



def ML(weight,surface):
	
	# Step 1 & 2                    1 -  rough     0 - smooth

	Features = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]


	Labels = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]     # 1 = tennis    2 = cricket



	# Step 3

	dobj = tree.DecisionTreeClassifier()			



	# Step 4

	dobj.fit(Features,Labels)						


	# Step 5

	result = dobj.predict([[weight,surface]])					

	if result == 1:

		print("Ball is Tennis")

	else:
		
		print("Ball is cricket")		




def main():


	print("************************* SUPERVISED  MACHINE  LEARNING  **************************")

	weight = int(input("Enter the weight:"))

	surface = input("Enter the surface:")

	if surface.lower() == "rough":

		surface = 1
		

	elif surface.lower() == "smooth":

		surface = 0

	else:
		print("Invalid Input")
		return


	ML(weight,surface)			


if __name__ == '__main__':
	main()