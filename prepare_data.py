from sklearn.cross_validation import StratifiedKFold
import numpy as np


def prepare_data():

	X = np.array([[1, 2], [3, 4], [5, 6], [7, 8] , [9, 10] , [11, 12] , [13, 14] , [15, 16] , [17, 18] , [19, 20]])
	y = np.array([0, 0, 1, 1 ,0 , 1 , 0, 1 ,0 , 1])

	print('data dimension is %s ' %str(X.shape))

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

		print('shape of array is %s ' %str(small_split_index.shape))

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

	print('test index is %s ' %str(test_indices))
	print('validation index are %s ' %str(validation_indices))
	print('train index is %s ' %str(train_indices))
	
	test_data = X[test_indices] 
	test_labels = y[test_indices]
	validation_data = X[validation_indices]
	validation_labels = y[validation_indices]
	train_data = X[train_indices]
	train_labels = y[train_indices]

	print('test data is %s ' %str(test_data))
	print('test labels are %s ' %str(test_labels))
	print('validation labels are %s ' %str(validation_data))
	print('validation data is %s ' %str(validation_labels))
	print('train data is %s ' %str(train_data))
	print('train labels are %s ' %str(train_labels))	


if __name__ == '__main__':
	prepare_data()
	
