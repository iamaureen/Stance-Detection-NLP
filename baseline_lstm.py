import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


#filename
filename = "data/emergent_without_null.csv"
fields = ['claim','claim_label','body','page_headline','page_position']

#read specific column
df = pd.read_csv(filename, usecols=fields)

#relabaling
df.loc[df['page_position'] == 'for', 'page_position'] = 0
df.loc[df['page_position'] == 'against', 'page_position'] = 1
df.loc[df['page_position'] == 'observing', 'page_position'] = 2
df.loc[df['page_position'] == 'ignoring', 'page_position'] = 3


df = shuffle(df)

#define models
models = [
    LogisticRegression(random_state=0, solver='lbfgs'), #add multiple models here if needed
]

# dummy classifier - majority-vote baseline
dummy = DummyClassifier(strategy="most_frequent") #most_frequent

# basic bag of words
#http://www.insightsbot.com/blog/R8fu5/bag-of-words-algorithm-in-python-introduction
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = "english")
features_bow_claim = vectorizer.fit_transform(df.claim).toarray()
features_bow_body = vectorizer.fit_transform(df.body).toarray()
labels = df.page_position
labels=labels.astype('int')
print('features_bow_claim',features_bow_claim.shape)
print('features_bow_body',features_bow_body.shape)

# concataned two bag  of words features
features_bow_combined = np.concatenate((features_bow_claim, features_bow_body), axis = 1)
print('features_bow_combined',features_bow_combined.shape)



print("---------80-20 training----------")
# 80-20 training
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features_bow_combined, labels, df.index, test_size=0.2, random_state=0)
models[0].fit(X_train, y_train)
y_pred = models[0].predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy on the current dataset(80-20 rule): {:.2f}".format(acc*100))
print("F1 Score", f1_score(y_test, y_pred, average="macro"))

# dummy classifier train
dummy.fit(X_train, y_train)

dummy_y_pred = dummy.predict(X_test)
dummy_acc = accuracy_score(y_test, dummy_y_pred)


print("Accuracy on the current dataset (baseline): {:.2f}".format(dummy_acc*100))
print("F1 Score", f1_score(y_test, dummy_y_pred, average="macro"))


print("---------cross validation----------")

CV = 5

accuracies = cross_val_score(models[0], features_bow_combined, labels, scoring='accuracy', cv=CV)
print("Accuracy on the current dataset(cross validation rule(logistic)): {:.2f}".format(accuracies.mean()*100))

accuracies = cross_val_score(models[0], features_bow_combined, labels, scoring='f1_macro', cv=CV)
print("F1 Measure on the current dataset(cross validation rule(logistic)): {:.2f}".format(accuracies.mean()))

accuracies = cross_val_score(DummyClassifier(strategy="most_frequent"), features_bow_combined, labels, scoring='accuracy', cv=CV)
print("Accuracy on the current dataset(cross validation rule (dummy)): {:.2f}".format(accuracies.mean()*100))

accuracies = cross_val_score(DummyClassifier(strategy="most_frequent"), features_bow_combined, labels, scoring='f1_macro', cv=CV)
print("F1 Measure on the current dataset(cross validation rule (dummy)): {:.2f}".format(accuracies.mean()))
