from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from sklearn.dummy import DummyClassifier


import pandas as pd

import merge_lstm

#get random data from data set
filename = "/Users/denizsonmez/Documents/distinct_entries_final.csv"
fields = ['claim','claim_label','body','page_headline','page_position']

df = pd.read_csv(filename, usecols=fields)

df = df.sample(n=500)

claim_content = df['claim'];
headline_content = df['page_headline'];

df.loc[df['claim_label'] == 'TRUE', 'claim_label'] = 0
df.loc[df['claim_label'] == 'FALSE', 'claim_label'] = 1
df.loc[df['claim_label'] == 'Unverified', 'claim_label'] = 2
labels = df.claim_label
labels=labels.astype('int')

#get max length claim
claim_field_length = df.claim.astype(str).map(len)
max_length_str = df.loc[claim_field_length.idxmax(), 'claim'];
max_length_claim = len(max_length_str.split());


#get max length headline
headline_field_length = df.page_headline.astype(str).map(len)
max_length_str = df.loc[headline_field_length.idxmax(), 'page_headline'];
max_length_headline= len(max_length_str.split());
print("Max headline length from fake news class:")
print(max_length_headline)

#get the two tokenizer for claim and body, also get the intermediate value from the training current dimension 200
claim_tokenizer, headline_tokenizer, final_model = merge_lstm.combine_lstm()


test_claim_seq = claim_tokenizer.texts_to_sequences(claim_content)
test_claim_padded = pad_sequences(test_claim_seq, maxlen=max_length_claim)

test_headline_seq = headline_tokenizer.texts_to_sequences(headline_content)
test_headline_padded = pad_sequences(test_headline_seq, maxlen=max_length_headline)


layer_name = 'concatenate_1'
intermediate_layer_model = Model(inputs=final_model.input,outputs=final_model.get_layer(layer_name).output)
pred = intermediate_layer_model.predict([test_claim_padded, test_headline_padded])
print(pred.shape)


logistic_model = LogisticRegression(random_state=0, solver='lbfgs') #add multiple models here if needed

# 80-20 training
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(pred, labels, df.index, test_size=0.2, random_state=0)
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Logistic Regression: Accuracy on the current dataset(80-20 rule): {:.2f}".format(acc*100))
print("Logistic Regression: F1 Score", f1_score(y_test, y_pred, average="macro"))

# majority vote
# dummy = DummyClassifier(strategy="most_frequent") #most_frequent
# dummy.fit(X_train, y_train)
#
# dummy_y_pred = dummy.predict(X_test)
# dummy_acc = accuracy_score(y_test, dummy_y_pred)
# print("Majority Vote: Accuracy on the current dataset(80-20 rule): {:.2f}".format(dummy_acc*100))
# print("Majority Vote: F1 Score", f1_score(y_test, dummy_y_pred, average="macro"))