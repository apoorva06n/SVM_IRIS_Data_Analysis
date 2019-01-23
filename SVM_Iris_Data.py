import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.grid_search import GridSearchCV

def main():
	#set default grid style for seaborn
	sns.set()
	#import iris dataset
	iris = sns.load_dataset('iris')
	#pairplot to differentiate different species
	sns.pairplot(data=iris,hue='species',palette='Dark2')
	plt.show()
	#create training and testing set by spliting dataset
	X=iris.drop('species',axis=1)
	y=iris['species']
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)
	#create a dictionary for parameters like C and gamma
	param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001]}
	#Create a grid object and fit it to the training data
	grid_obj= GridSearchCV(SVC(),param_grid,verbose=2)
	grid_obj.fit(X_train,y_train)
	pred = grid_obj.predict(X_test)
	print("\n---------- Prediction evaluation using Confusion Matrix -----------\n")
	print(confusion_matrix(y_test,pred))
	print("\n---------- Prediction evaluation using Classification Report -----------\n")
	print(classification_report(y_test,pred))

if __name__ == '__main__':
	main()