from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Activation, Dense, concatenate
from tensorflow.python.keras.layers import add
import numpy as np

import claim_lstm_model as cm
import body_lstm_model as bm
import prepare_data as pd

import os

os.environ["CUDA_VISIBLE_DEVICES"]="3"

padded_claim_content, claim_model, labels, test_claim_padded = cm.get_claim_lstm_model();
'''
This line is for the case that we were using body content and not haedline but it seems that body we nee to use
body headline because body is the same even for different news from different domains
'''
#padded_body_content, body_model, test_body_padded = bm.get_body_lstm_model();

padded_headline_content, body_model, test_body_padded = bm.get_body_lstm_model();

print('strat preparing data split')

padded_claim_content = np.array(padded_claim_content)
padded_headline_content = np.array(padded_headline_content)
labels = np.array(labels)

pd.prepare_data(padded_claim_content, padded_headline_content, labels)

print('finish preparing data split')

# Note: test_claim_padded is just one sample test and we need to prepare the test, validation and train 
# by the prepare_data function which we have prepared


# #https://stackoverflow.com/questions/51075666/how-to-implement-merge-from-keras-layers
# merged_output = add([claim_model.output, body_model.output])
merged_output = concatenate([claim_model.output, body_model.output])

model_combined = Sequential()
model_combined.add(Activation('relu'))
model_combined.add(Dense(256))
model_combined.add(Activation('relu'))
model_combined.add(Dense(4))
model_combined.add(Activation('softmax'))

final_model = Model([claim_model.input, body_model.input], model_combined(merged_output))

final_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# print(final_model.summary())

print('dimension of padded_claim_content is %s' %str(padded_claim_content.shape))

final_model.fit([padded_claim_content, padded_headline_content], label, validation_split=0.4, epochs=10)

print(final_model.summary())

# get the intermediate concatenation layer vector with dimension 200
layer_name = 'concatenate_1'
intermediate_layer_model = Model(inputs=final_model.input,outputs=final_model.get_layer(layer_name).output)
pred = intermediate_layer_model.predict([test_claim_padded,test_body_padded])

print(pred[1])


#print('dimension of padded_claim_content is %s' %str(padded_claim_content.shape))

loss, accuracy = final_model.evaluate([padded_claim_content, padded_headline_content], label, verbose=0)
print('Accuracy: %f' % (accuracy*100))

#testing. using a different sentence for texts_to_sequences (follow the stackoverflow link above)

#prediction in terms of probability and output the predicted class
# pred = final_model.predict([test_claim_padded,test_body_padded])
# labels = ['0', '1', '2', '3']
# print(pred, labels[np.argmax(pred)])







