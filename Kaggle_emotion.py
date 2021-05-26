import os
import time
import string, re
import pickle, nltk

from operator import itemgetter
import pandas as pd
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag , word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.cross_validation import train_test_split as tts
from sklearn.model_selection import train_test_split as tts
from sklearn.svm import LinearSVC


#################################Chargement de data############################################
df = pd.read_csv('emotions_final.csv')
print(df.Emotion.value_counts())

##################### Preprocessing data / nettoyer les données #######################
##### Lowercasing
texts = df['Text'].str.lower()

# ##### Remove special chars
# texts = texts.str.replace(r"(http|@)\S+", "")
# texts = texts.str.replace(r"::", ": :blush:")
# texts = texts.str.replace(r"’", "'")
# texts = texts.str.replace(r"[^a-z':_]", " ")

# ##### Remove repetitions
# pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
# texts = texts.str.replace(pattern, r"\1")

# ##### Transform short negation form
# texts = texts.str.replace(r"(can't|cannot)", 'can not')
# texts = texts.str.replace(r"n't", ' not')

##### Remove stop words
stopwords = nltk.corpus.stopwords.words('english')
stopwords.remove('not')
stopwords.remove('nor')
stopwords.remove('no')
texts = texts.apply(
  lambda x: ' '.join([word for word in x.split() if word not in stopwords])
)

##### Lemmatize
lemmatizer=WordNetLemmatizer()
def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(w) for w in word_tokenize(text)])

Lemmatizedtexts=texts.apply(lemmatize_text)
#print(Lemmatizedtexts)

# vect = TfidfVectorizer(stop_words='english',analyzer='word',ngram_range=(1,2))
# tfidf_mat = vect.fit_transform(Lemmatizedtexts)
# feature_names = vect.get_feature_names() #to get the nams of the tokens
# dense = tfidf_mat.todense() #convert sparse matrix to numpy array
# denselist = dense.tolist() #convert array to list
# df2 = pd.DataFrame(denselist,columns=feature_names) #convert to dataframe
# print(df.head())
# print('###########################################')


##### Split data
# X_train, X_test, y_train, y_test = tts(Lemmatizedtexts,df.Emotion,stratify=df.Emotion,test_size=0.2, random_state=42)
# clf = LinearSVC()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print('F1 score : ', f1_score(y_test, y_pred))