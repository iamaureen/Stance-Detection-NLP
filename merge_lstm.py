from keras.models import Model, Sequential
from keras.layers import merge, Activation, Dense, concatenate
from keras.layers import add

import claim_lstm_model as cm
import body_lstm_model as bm


padded_claim_content, claim_model, label = cm.get_claim_lstm_model();
padded_body_content, body_model = bm.get_body_lstm_model();

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

final_model.fit([padded_claim_content, padded_body_content], label, validation_split=0.4, epochs=10)

print(final_model.summary())

loss, accuracy = final_model.evaluate([padded_claim_content, padded_body_content], label, verbose=0)
print('Accuracy: %f' % (accuracy*100))







