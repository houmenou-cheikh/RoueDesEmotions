import pandas as pd
import string

#for machine learning - classification
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#for visualization
import seaborn as sns
import matplotlib.pyplot as plt

#for NLP
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk import ngrams

#les Data
df1 = pd.read_csv("/home/houmenou/Documents/LaPlateforme_Jobs/RoueDesEmotions/YouTube-Spam-Collection-v1/Youtube01-Psy.csv")
df2 = pd.read_csv("/home/houmenou/Documents/LaPlateforme_Jobs/RoueDesEmotions/YouTube-Spam-Collection-v1/Youtube02-KatyPerry.csv")
df3 = pd.read_csv("/home/houmenou/Documents/LaPlateforme_Jobs/RoueDesEmotions/YouTube-Spam-Collection-v1/Youtube03-LMFAO.csv")
df4 = pd.read_csv("/home/houmenou/Documents/LaPlateforme_Jobs/RoueDesEmotions/YouTube-Spam-Collection-v1/Youtube04-Eminem.csv")
df5 = pd.read_csv("/home/houmenou/Documents/LaPlateforme_Jobs/RoueDesEmotions/YouTube-Spam-Collection-v1/Youtube05-Shakira.csv")

df = pd.concat([df1,df2,df3,df4,df5])
df = df[['CONTENT','CLASS']].copy()

#exploration de data
print(df.CLASS.value_counts())
print("#############################################")
print(df.sample(5))

#Preprocessing des données
print("#############################################")

## Normalization:
#Normaliser le texte signifie le mettre à la même casse, souvent tout en minuscule.

phrase = "j'ai BEAUCOUP mangé"
print(phrase.lower())

print("#############################################")

##Tokénization

"""Passons à la Tokénization ! C’est un procédé très simple qui divise une chaîne de caractère en tokens, 
c’est-à-dire des éléments atomiques de la chaîne. Un token n’est pas forcément un mot, ce peut être par 
exemple un signe de ponctuation. NLTK fournit plusieurs types de tokénization, comme la tokénization par 
mot ou par phrase. En effet, si on considère que la fin d’une phrase correspond à un point puis un espace 
puis une majuscule, nous aurions du mal avec les acronymes ou les titres (Dr. Intel). Ce que NLTK propose, 
ce sont donc des tokenizers déjà entraînés sur un set de documents (ou corpus). Dans notre cas, nous n’avons 
besoin que du tokenizer de mots. Exemple:"""

print(word_tokenize('I would like an orangenjuice, and a sandwich!'))
print('###########################################')
##Suppression des stop words

"""Vient ensuite l’étape de suppression des stopwords qui est cruciale, car elle va enlever dans le texte tous 
les mots qui n’ont que peu d’intérêt sémantique. Les stopwords sont en effet tous les mots les plus courants 
d’une langue (déterminants, pronoms, etc..). NLTK dispose d’une liste de stopwords en anglais (ou dans d’autres langues). Exemple:"""

stopW = stopwords.words('english')
print('il y a {} stopwords'.format(len(stopW)))
print('les 10 premiers sont {} \n'.format(stopW[:10]))

exclude = set(string.punctuation)
tokens = word_tokenize('I would like an orange juice, and a sandwich!')
print('input tokens: {}'.format(tokens))
stopW.extend(exclude) #we add the punctuation to the previous stop words list
tokens_without_stopW = [word for word in tokens if word not in stopW]
print('output tokens: {}'.format(tokens_without_stopW))

print('###########################################')

##Stemming et Lemmatization
""" Ces deux méthodes sont très couramment utilisées dans le traitement du langage naturel car permettent de 
représenter sous un même mot plusieurs dérivées du mot. Dans le cas du Stemming, nous allons uniquement garder
le radical du mot (ex : dormir, dortoir et dors deviendront dor). La lemmatization, moins radicale 😉, va laisser 
au mot un sens sémantique mais va éliminer le genre ou le pluriel par exemple. """

lemma = WordNetLemmatizer()
text = word_tokenize('The girls wanted to play with thier parents')
print([lemma.lemmatize(word) for word in text])

""" Nous pouvons maintenant appliquer en un coup la lemmatization et la normalisation à notre dataframe. 
Ici, nous appliquons la tokénization dans le but de faire la lemmatization, mais nous rejoignons les tokens 
(avec la fonction join) car nous allons ici avoir besoin de cette forme plus tard, tout dépend de l’application."""

def lemmatizer(sent):
    tokens = word_tokenize(sent.lower())
    tokens = [lemma.lemmatize(lemma.lemmatize(lemma.lemmatize(w,'v'), 'n'), 'a') for w in tokens]
    return ' '.join(tokens)

df['CONTENT'] = df.CONTENT.apply(lambda sent: lemmatizer(sent))
#print(df['CONTENT'])

""" Dans la pratique, le Stemming est employé surtout pour effectuer des recherches sur un grand nombre de document (ex : moteur de recherche), pour le reste, la lemmatization est souvent préférée.

Ces méthodes sont employées pour deux raisons :

-Donner le même sens à des mots très proches mais d’un genre différent (ou éliminer le pluriel, etc…)
-Réduire la sparsité des matrices utilisées dans les algorithmes (voir partie suivante sur TFIDF)"""

print('###########################################')

## N-grams

""" Les n-grams sont tous simplement des suites de mots présents dans le texte. 
Ce que nous traitions jusqu’à présent étaient uniquement des unigrammes, nous 
pouvons ensuite rajouter des bigrammes ou même des trigrammes. Un bigramme est un
 couple de mot qui se suivent dans le texte, nous pouvons les trouver facilement 
 grâce à NLTK :"""

tokens = word_tokenize('The girls wanted to play with their pareents')
bigrams = ngrams(tokens,2)
for words in bigrams:
    print(words)

#Classification

"""Vous le savez peut-être mais les algorithmes n’aiment pas les mots… Heureusement pour nous, 
il existe des méthodes simples permettant de convertir un document en une matrice de mot. 
Ces matrices étant souvent creuses (sparse en anglais), c’est-à-dire pleines de 0 avec peu de 
valeurs, la lemmatization aide à réduire leurs tailles. Afin de convertir ces phrases en 
matrice, nous allons voir une méthode que l’on appelle TFIDF (Term Frequency – Inverse
 Document Frequency). """

## TFIDF
""" TFIDF est une approche bag-of-words (bow) permettant de représenter les mots d’un document à l’aide 
d’une matrice de nombres. Le terme bow signifie que l’ordre des mots dans la phrase n’est pas pris en compte,
contrairement à des approches plus poussées de Deep Learning"""

vect = TfidfVectorizer(stop_words='english',analyzer='word',ngram_range=(1,2))
tfidf_mat = vect.fit_transform(df.CONTENT)
feature_names = vect.get_feature_names() #to get the nams of the tokens
dense = tfidf_mat.todense() #convert sparse matrix to numpy array
denselist = dense.tolist() #convert array to list
df2 = pd.DataFrame(denselist,columns=feature_names) #convert to dataframe
print(df2.head())
print('###########################################')


#Support Vector Machine

""" Une machine à vecteurs de support (SVM) est un algorithme permettant de réaliser des tâches 
de classification ou de régression, très en vogue il y a quelques années mais depuis largement 
surpassé par les réseaux de neurones profonds. Néanmoins, il fonctionne bien sur des données 
textuelles. Son principe est de séparer au maximum les exemples tirés des différentes classes,
 on le qualifie de hard margin classifier """

X_train, X_test, y_train, y_test = train_test_split(df2, df.CLASS, test_size = 0.3)
clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('F1 score : ', f1_score(y_test, y_pred))

#matrice de confusion concluante
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.ylabel('True classes')
plt.xlabel('Predicted classes')
plt.show()

#Pour rappel, une matrice de classification nous renseigne sur le nombre de spam/ham 
#correctement classés ou bien sur ceux qui ont été mal classés.

""" Le score F1 est une métrique très utile dans les tâches de classification, nous indiquant 
à la fois la précision et le recall du modèle, qui se calculent grâce à la matrice de confusion
 ci-dessus et dont vous pourrez trouver le détail sur wikipédia 😉"""