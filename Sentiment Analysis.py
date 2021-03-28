# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
# For visualizations
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer ,CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report

dataFrame = pd.read_csv(
                        "training.1600000.processed.noemoticon.csv",
                        encoding = 'latin-1')

#Dataset has not columns names

dataFrame.columns = ["sentiment","id","data","Query","User","text"]

#we need only sentiment and text columns so we will drop them

dataFrame = dataFrame[["sentiment","text"]]

#pozitive tweets has 4 sentiments and negatives have 0
#Because of execution time i decreased size of dataframe
pozitive_data = dataFrame[dataFrame['sentiment']== 4].iloc[:40000]


negative_data = dataFrame[dataFrame['sentiment']== 0].iloc[:40000]

dataFrame = pd.concat([pozitive_data,negative_data],axis = 0)

ax = dataFrame.groupby('sentiment').count().plot(kind='bar', title='Distribution of data',
                                               legend=False)
ax.set_xticklabels(['Negative','Positive'], rotation=0)

#Word cloud for all words but for not mixing i set max_words as 1000

from wordcloud import WordCloud
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(dataFrame["text"]))
plt.imshow(wc)

#Word cloud for positive texts but for not mixing i set max_words as 1000 also as you can see good, thank, lol, love is most common words in positive tweets

from wordcloud import WordCloud
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(pozitive_data["text"]))
plt.imshow(wc)

#Word cloud for negative tweets but for not mixing i set max_words as 1000

from wordcloud import WordCloud
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(negative_data["text"]))
plt.imshow(wc)

#Make lower case
dataFrame["text"] = dataFrame["text"].apply(lambda x: " ".join(x.lower() for x in x.split()))

#Clean Url

dataFrame['text'] = dataFrame['text'].str.replace("((www\.[^\s])+|(https?://[^\s]+))",'')

#Clean User Nicknames

dataFrame['text'] = dataFrame['text'].str.replace(r'@[A-Za-z0-9_]+', "") 

#deleting symbols
dataFrame['text'] = dataFrame['text'].str.replace("[^\w\s]", " ") 

#clean numbers
dataFrame['text'] = dataFrame['text'].str.replace("[\d]", " ") 

#calling stopwords with natural language tool kit

nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words("english")

#I added 'im' and 'u' after checking it is unnecessary and clean them
sw.append("im")
sw.append("u")
dataFrame["text"]= dataFrame["text"].apply(lambda x : " ".join(x for x in x.split() if x not in sw))


#Tokenize texts
dataFrame["text"] = dataFrame["text"].apply(lambda x : x.split())


#Steeming
from nltk.stem.porter import * 
stemmer = PorterStemmer() 
dataFrame['text'] = dataFrame['text'].apply(lambda x: [stemmer.stem(i) for i in x])

#detokenizing
dataFrame['text'] = dataFrame['text'].apply(lambda x: ' '.join([w for w in x]))

#Vectorize dataframe

count_vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", 
                stop_words='english', analyzer='word') 
cv = count_vectorizer.fit_transform(dataFrame['text'])

#i mention at report and i showed it here
count_vectorizer.get_feature_names()[0:5]

#genereate train sets
X_train,X_test,y_train,y_test = train_test_split(cv,dataFrame['sentiment'] , test_size=.2,stratify=dataFrame['sentiment'], random_state=2)

import xgboost
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics ,decomposition, ensemble , svm
from sklearn.metrics import accuracy_score
import time

from warnings import filterwarnings
filterwarnings('ignore')

"""#Logistic Regression"""

t0 = time.time()
lr = LogisticRegression(max_iter = 2000)
lr.fit(X_train,y_train)
prediction_lr = lr.predict(X_test)
t1 = time.time()
print("Accuracy of Logistic Regression = " , accuracy_score(prediction_lr,y_test), "\nProcess is taken ",round((t1-t0),2) , "seconds. ")

"""#Naive Bayes"""

t0 = time.time()

nb = naive_bayes.MultinomialNB()
nb.fit(X_train,y_train)
prediction_NB = nb.predict(X_test)
t1 = time.time()
print("Accuracy of Naive Bayes  = ", accuracy_score(prediction_NB,y_test), "\nProcess is taken ",round((t1-t0),2) , " seconds. ")

"""#Random Forests"""

t0 = time.time()
rf = ensemble.RandomForestClassifier()
rf_model = rf.fit(X_train,y_train)
rf_model = rf_model.predict(X_test)
t1 = time.time()
print("Accuracy of Random Forests = ",accuracy_score(rf_model,y_test), "\nProcess is taken ",round((t1-t0),2) , " seconds. ")

"""## XGBoost"""

t0 = time.time()
xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(X_train,y_train)
prediction_xgb = xgb_model.predict(X_test)
t1 = time.time()
print("Accuracy of XGBoost  = " , accuracy_score(prediction_xgb,y_test), "\nProcess is taken " ,round((t1-t0),2) ," seconds. ")
