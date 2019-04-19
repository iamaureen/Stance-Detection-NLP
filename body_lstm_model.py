import pandas as pd
import numpy as np
from numpy import asarray
from numpy import zeros
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from tensorflow.python.keras import losses

def get_body_lstm_model():

	#filename
	filename = "data/emergent_without_null.csv"


	fields = ['claim','claim_label','body','page_headline','page_position']

	df = pd.read_csv(filename, usecols=fields)

	#todo remove duplicate claim rows and corresponding page position too
	body_content = df['body'];
	print(len(body_content));

	#labels
	page_position = df['page_position'];
	print(len(page_position));

	df.loc[df['page_position'] == 'for', 'page_position'] = 0
	df.loc[df['page_position'] == 'against', 'page_position'] = 1
	df.loc[df['page_position'] == 'observing', 'page_position'] = 2
	df.loc[df['page_position'] == 'ignoring', 'page_position'] = 3
	df.loc[df['page_position'] == 'nan', 'page_position'] = 5


	page_position_list = df.page_position.unique()
	print(page_position_list)

	# prepare tokenizer
	t = Tokenizer()
	t.fit_on_texts(body_content)
	vocab_size = len(t.word_index) + 1

	# integer encode the documents
	encoded_docs = t.texts_to_sequences(body_content)
	# print(encoded_docs)

	# https://stackoverflow.com/questions/34757703/how-to-get-the-longest-length-string-integer-float-from-a-pandas-column-when-the?rq=1
	# get the max length of the body column
	body_field_length = df.body.astype(str).map(len)
	# print (type(df.loc[claim_field_length.idxmax(), 'claim']))
	# print (type(df.loc[claim_field_length.idxmin(), 'claim']))

	max_length_str = df.loc[body_field_length.idxmax(), 'body'];
	max_length = len(max_length_str.split());

	padded_body_content = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
	print(padded_body_content)
	# load the whole embedding into memory
	embeddings_index = dict()
	f = open('data/glove.6B.100d.txt')
	for line in f:
		values = line.split()
		word = values[0]
		coefs = asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	print('Loaded %s word vectors.' % len(embeddings_index))
	# create a weight matrix for words in training docs
	embedding_matrix = zeros((vocab_size, 100))
	for word, i in t.word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	## create the LSTM model
	model_lstm = Sequential()
	model_lstm.add(Embedding(vocab_size, 100, input_length=max_length, weights=[embedding_matrix], trainable=False))
	# model_lstm.add(Dropout(0.2))
	# model_lstm.add(Conv1D(64, 5, activation='relu'))
	# model_lstm.add(MaxPooling1D(pool_size=4))
	model_lstm.add(LSTM(100))
	# activation function = softmax instead of sigmoid since multiclass, sigmoid is for binary class
	# First parameter of Dense is the unit = number of labels  https://keras.io/layers/core/
	#model_lstm.add(Dense(4, activation='relu'))
	# use sparse_categorical_crossentropy instead of binary_crossentropy, because multi class; also the class labels
	# needs to start from 0.
	#model_lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	## Fit train data
	#model_lstm.fit(padded_body_content, page_position, validation_split=0.4, epochs = 1)

	# evaluate the model
	# loss, accuracy = model_lstm.evaluate(padded_body_content, page_position, verbose=0)
	# print('Accuracy: %f' % (accuracy*100))

	# testing. using a different sentence for texts_to_sequences (follow the stackoverflow link above)
	test_txt = ["Regular fast food eating linked to fertility issues in women",
				"On Feb. 15 Reddit user Electrorocket posted a link to a YouTube video that allegedly compares the speed of a regular Seinfeld episode with one broadcast on TBS. The video has been online since Sept. 2013, but the post on Reddit (reddit.com/r/television/comments/2vwgk9/tbs_speeds_up_seinfeld_75_to_gain_an_extra_2) attracted new interest and media coverage. The video was also posted to AVSForum.com around the same time, with more details from the user. As of now, it's not clear whether the claim is true, though it has been confirmed that other stations speed up reruns. We will seek confirmation from TBS."]
	seq = t.texts_to_sequences(test_txt)
	test_body_padded = pad_sequences(seq, maxlen=max_length)

	return padded_body_content, model_lstm,test_body_padded;
