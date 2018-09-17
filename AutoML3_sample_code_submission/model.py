'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
import random
from sklearn.ensemble import GradientBoostingClassifier
class Model:
	def __init__(self):
		'''
		This constructor is supposed to initialize data members.
		Use triple quotes for function documentation. 
		'''
		pass

	def fit(self, X, y):
		'''
		This function should train the model parameters.
		Here we do nothing in this example...
		Args:
			X: Training data matrix of dim num_train_samples * num_feat.
			y: Training label matrix of dim num_train_samples * num_labels.
		Both inputs are numpy arrays.
		If fit is called multiple times on incremental data (train, test1, test2, etc.)
		you should warm-start your training from the pre-trained model. Past data will
		NOT be available for re-training.
		'''
		pass

	def predict(self, X):
		'''
		This function should provide predictions of labels on (test) data.
		Here we just return random values...
		Make sure that the predicted values are in the correct format for the scoring
		metric. For example, binary classification problems often expect predictions
		in the form of a discriminant value (if the area under the ROC curve it the metric)
		rather that predictions of the class labels themselves. 
		The function predict eventually can return probabilities or continuous values.
		'''
		num_test_samples = X.shape[0]
		y= np.random.rand(num_test_samples)
		return y

	def save(self, path="./"):
		pickle.dump(self, open(path + '_model.pickle', "w"))

	def load(self, path="./"):
		modelfile = path + '_model.pickle'
		if isfile(modelfile):
			with open(modelfile) as f:
				self = pickle.load(f)
			print("Model reloaded from: " + modelfile)
		return self
