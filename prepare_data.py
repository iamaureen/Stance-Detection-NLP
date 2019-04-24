from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import numpy as np



def prepare_data(claims, headlines, labels):


	print('inital dimension of claim array is %s ' %str(claims.shape))

	print('inital dimension of headline array is %s ' %str(headlines.shape))

	print('inital dimension of labels array is %s ' %str(labels.shape))

	# in the peject we need to divide our data to 3 separate parts. 
	# 1) Train 0.8
	# 2) Test 0.1
	# 3) Validation 0.1

	# to implement this, we can divide to 10 parts and use, 1 part as test, 1 part as train and one part as validation

	# verified: by using the following function, we keep the dataset balanced
	label_encoder = LabelEncoder()
	labels = label_encoder.fit_transform(labels)
	cv = StratifiedKFold(n_splits = 10)

	validation_indices = [] 
	test_indices = []
	train_indices = []

	index = 0


	'''
	First we need to prepare the indices for test, validation and train

	'''

	for large_split_index, small_split_index in cv.split(claims, labels):
		print("here")
		#print('shape of array is %s ' %str(small_split_index.shape))

		if index == 0:
			test_indices = small_split_index

		
		elif index == 1:
			validation_indices = small_split_index
		
		else:
			if train_indices == []:
				train_indices = small_split_index 
			else:

				train_indices = np.concatenate((train_indices , small_split_index)) 
		
		index += 1
	
	'''
	After preparing the indices for data, we sould prepare the data points
	'''


	test_claims = claims[test_indices]
	validation_claims = claims[validation_indices]
	train_claims = claims[train_indices]

	test_headlines = headlines[test_indices]
	validation_headlines = headlines[validation_indices]
	train_headlines = headlines[train_indices]

	test_labels = labels[test_indices]
	validation_labels = labels[validation_indices]
	train_labels = labels[train_indices]
	
	print('Dimension of test_claims is %s ' %str(test_claims.shape)) 
	print('Dimension of validation_claims is %s '  %str(validation_claims.shape))
	print('Dimension of train_claims is %s ' %str(train_claims.shape))

	print('Dimension of test_healines is %s '  %str(test_headlines.shape))
	print('Dimension of validation_headlines is %s ' %str(validation_headlines.shape))
	print('Dimension of train_headlines is %s ' %str(train_headlines.shape))

	print('Dimension of test_labels is %s ' %str(test_labels.shape))
	print('Dimension of validation_labels is %s '  %str(validation_labels.shape))
	print('Dimension of train_labels is %s '  %str(train_labels.shape))

	return test_claims , validation_claims , train_claims , test_headlines , validation_headlines , train_headlines , test_labels , validation_labels , train_labels


	# The way that I am returing is veryyyy bad. CHANGE it to a class
	#return test_data, test_labels, validation_data, validation_labels, train_data, train_labels
if __name__ == '__main__':
	prepare_data()
	
