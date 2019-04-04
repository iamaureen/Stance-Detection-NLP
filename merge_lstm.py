from keras.models import Sequential
from keras.layers import Add, Activation, Dense

import claim_lstm_model as cm
import body_lstm_model as bm


padded_claim_content, claim_model, label = cm.get_claim_lstm_model();
padded_body_content, body_model = bm.get_body_lstm_model();

merged_model = Sequential()
merged_model.add(Add([claim_model, body_model], mode='concat'))
merged_model.add(Dense(4, activation='softmax'))
merged_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
merged_model.fit([padded_claim_content, padded_body_content], label, validation_split=0.4)

loss, accuracy = merged_model.evaluate(padded_claim_content, padded_body_content, label, verbose=0)
print('Accuracy: %f' % (accuracy*100))






