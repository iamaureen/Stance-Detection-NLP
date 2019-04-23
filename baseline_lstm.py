import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from scipy.stats import ttest_rel

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


#filename
filename =  "/Users/denizsonmez/Documents/distinct_entries_final.csv"
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
features_bow_headline = vectorizer.fit_transform(df.page_headline).toarray()
print('features_bow_claim',features_bow_claim.shape)
print('features_bow_headline',features_bow_headline.shape)

# concataned two bag  of words features
features_bow_combined = np.concatenate((features_bow_claim, features_bow_headline), axis = 1)
print('features_bow_combined',features_bow_combined.shape)

#tf-idf
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l1', encoding='latin-1', ngram_range=(1,3), stop_words='english')
features_tfidf_claim = tfidf.fit_transform(df.claim).toarray();
features_tfidf_headline = tfidf.fit_transform(df.page_headline).toarray();
labels = df.page_position
labels=labels.astype('int')
print('features_bow_claim',features_tfidf_claim.shape)
print('features_bow_headline',features_tfidf_headline.shape)

# concataned two bag  of words features
features_tfidf_combined = np.concatenate((features_tfidf_claim, features_tfidf_headline), axis = 1)
print('features_bow_combined',features_bow_combined.shape)



print("---------80-20 training----------")
# 80-20 training
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features_tfidf_combined, labels, df.index, test_size=0.2, random_state=0)
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

accuracies1 = cross_val_score(models[0], features_tfidf_combined, labels, scoring='accuracy', cv=CV)
print("Accuracy on the current dataset(cross validation rule(logistic)): {:.2f}".format(accuracies1.mean()*100))

accuracies = cross_val_score(models[0], features_tfidf_combined, labels, scoring='f1_macro', cv=CV)
print("F1 Measure on the current dataset(cross validation rule(logistic)): {:.2f}".format(accuracies.mean()))

accuracies2 = cross_val_score(DummyClassifier(strategy="most_frequent"), features_tfidf_combined, labels, scoring='accuracy', cv=CV)
print("Accuracy on the current dataset(cross validation rule (dummy)): {:.2f}".format(accuracies2.mean()*100))

accuracies = cross_val_score(DummyClassifier(strategy="most_frequent"), features_tfidf_combined, labels, scoring='f1_macro', cv=CV)
print("F1 Measure on the current dataset(cross validation rule (dummy)): {:.2f}".format(accuracies.mean()))

print("T test results comparing majority vote and logistic regression baselines")
print(ttest_rel(accuracies1, accuracies2))

