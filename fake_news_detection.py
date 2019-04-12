from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model

import pandas as pd

import merge_lstm

#get random data from data set
filename = "data/emergent_without_null.csv"
fields = ['claim','claim_label','body','page_headline','page_position']

df = pd.read_csv(filename, usecols=fields)

df = df.sample(n=500)

claim_content = df['body'];
body_content = df['claim'];

df.loc[df['claim_label'] == 'TRUE', 'claim_label'] = 0
df.loc[df['claim_label'] == 'FALSE', 'claim_label'] = 1
df.loc[df['claim_label'] == 'Unverified', 'claim_label'] = 2
labels = df.claim_label
labels=labels.astype('int')

#get max length claim
claim_field_length = df.claim.astype(str).map(len)
max_length_str = df.loc[claim_field_length.idxmax(), 'claim'];
max_length_claim = len(max_length_str.split());


#get max length body
body_field_length = df.body.astype(str).map(len)
max_length_str = df.loc[body_field_length.idxmax(), 'body'];
max_length_body = len(max_length_str.split());

#get the two tokenizer for claim and body, also get the intermediate value from the training current dimension 200
claim_tokenizer, body_tokenizer, final_model = merge_lstm.combine_lstm()


test_claim_seq = claim_tokenizer.texts_to_sequences(claim_content)
test_claim_padded = pad_sequences(test_claim_seq, maxlen=max_length_claim)

test_body_seq = claim_tokenizer.texts_to_sequences(claim_content)
test_body_padded = pad_sequences(test_body_seq, maxlen=max_length_body)


layer_name = 'concatenate_1'
intermediate_layer_model = Model(inputs=final_model.input,outputs=final_model.get_layer(layer_name).output)
pred = intermediate_layer_model.predict([test_claim_padded,test_body_padded])
print(pred.shape)


logistic_model = LogisticRegression(random_state=0, solver='lbfgs') #add multiple models here if needed

# 80-20 training
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(pred, labels, df.index, test_size=0.2, random_state=0)
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy on the current dataset(80-20 rule): {:.2f}".format(acc*100))
print("F1 Score", f1_score(y_test, y_pred, average="macro"))