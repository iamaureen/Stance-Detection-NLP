from keras.models import Model, Sequential
from keras.layers import merge, Activation, Dense, concatenate
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import model_from_yaml
from keras.utils import to_categorical



from keras.layers import add
import numpy as np

import claim_lstm_model as cm
import headline_lstm_model as bm
import prepare_data as pd


def combine_lstm():
    padded_claim_content, claim_model, labels, claim_tokenizer = cm.get_claim_lstm_model();
    padded_headline_content, headline_model, headline_tokenizer = bm.get_headline_lstm_model();

    padded_claim_content = np.array(padded_claim_content)
    padded_headline_content = np.array(padded_headline_content)
    labels = np.array(labels)
    # #https://stackoverflow.com/questions/51075666/how-to-implement-merge-from-keras-layers
    # merged_output = add([claim_model.output, body_model.output])
    test_claims, validation_claims, train_claims, test_headlines, validation_headlines, train_headlines, test_labels, validation_labels, train_labels = pd.prepare_data(
        padded_claim_content, padded_headline_content, labels)

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    validation_labels = to_categorical(validation_labels)

    # merged_output = concatenate([claim_model.output, headline_model.output])
    #
    # model_combined = Sequential()
    # model_combined.add(Activation('relu'))
    # model_combined.add(Dense(256))
    # model_combined.add(Activation('relu'))
    # model_combined.add(Dense(4))
    # model_combined.add(Activation('softmax'))
    #
    # final_model = Model([claim_model.input, headline_model.input], model_combined(merged_output))
    #
    # optimizer = Adam(lr=1e-5)
    # final_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    #
    # # print(final_model.summary())
    # weight_path = 'best_weight.h5'
    # checkpoint = ModelCheckpoint(weight_path, monitor='val_categorical_accuracy', verbose=2, save_best_only=True,
    #                              mode='max', period=1)
    #
    # # callbacks_list = [checkpoint]
    # tensorboad = TensorBoard(log_dir='./logs')
    #
    # callbacks_list = [checkpoint, tensorboad]
    # print("Train Label Shape")
    # print(train_labels.shape)
    # final_model.fit([train_claims, train_headlines], train_labels, validation_data=([test_claims, test_headlines], test_labels) , epochs=50 , callbacks = callbacks_list)
    #
    # print(final_model.summary())
    #
    # model_yaml = final_model.to_yaml()
    # with open("model.yaml", "w") as yaml_file:
    #     yaml_file.write(model_yaml)
    #
    # loss, accuracy = final_model.evaluate([validation_claims, validation_headlines], validation_labels, verbose=0)
    # print('Accuracy: %f' % (accuracy*100))

    final_model = model_from_yaml(open('model.yaml', 'r').read())
    final_model.load_weights('best_weight.h5')
    return claim_tokenizer, headline_tokenizer, final_model;







