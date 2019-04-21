from sklearn.cross_validation import StratifiedKFold
import numpy as np


def prepare_data_test():

	X = np.array([[1, 2], [3, 4], [5, 6], [7, 8] , [9, 10] , [11, 12] , [13, 14] , [15, 16] , [17, 18] , [19, 20]])
	y = np.array([0, 0, 1, 1 ,0 , 1 , 0, 1 ,0 , 1])

	#print('data dimension is %s ' %str(X.shape))

	# in the peject we need to divide our data to 3 separate parts. 
	# 1) Train 0.8
	# 2) Test 0.1
	# 3) Validation 0.1

	# to implement this, we can divide to 10 parts and use, 1 part as test, 1 part as train and one part as validation

	# verified: by using the following function, we keep the dataste balanced

	cv = StratifiedKFold(y, n_folds=5)

	validation_indices = [] 
	test_indices = []
	train_indices = []

	index = 0


	'''
	First we need to prepare the indices for test, validation and train

	'''

	for large_split_index, small_split_index in cv:

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

	test_data = X[test_indices] 
	test_labels = y[test_indices]
	validation_data = X[validation_indices]
	validation_labels = y[validation_indices]
	train_data = X[train_indices]
	train_labels = y[train_indices]


	'''
	print('test data is %s ' %str(test_data))
	print('test labels are %s ' %str(test_labels))
	print('validation labels are %s ' %str(validation_data))
	print('validation data is %s ' %str(validation_labels))
	print('train data is %s ' %str(train_data))
	print('train labels are %s ' %str(train_labels))	
	'''

	# The way that I am returing is veryyyy bad. CHANGE it to a class
	return test_data, test_labels, validation_data, validation_labels, train_data, train_labels

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

	cv = StratifiedKFold(labels, n_folds=10)

	validation_indices = [] 
	test_indices = []
	train_indices = []

	index = 0


	'''
	First we need to prepare the indices for test, validation and train

	'''

	for large_split_index, small_split_index in cv:

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
	
