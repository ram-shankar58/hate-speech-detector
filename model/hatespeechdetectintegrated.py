
# !pip install scikit-learn

import pandas as pd
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import seaborn
import string
import re
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

# !pip install nltk

# !pip install vaderSentiment

# !pip install textstat

import nltk
from nltk.stem.porter import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
import warnings

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

df=pd.read_csv('/content/labeled_data.csv')
df.head()

df.describe()

df.columns

tweets=df.tweet

#importing list of stopwords
# nltk.download('stopwords')
stopwords=nltk.corpus.stopwords.words("english")

exclusions=["#ff", "ff", "rt", "austrian painter"] #add any words you want to ignore
stopwords.extend(exclusions)

stemmer=PorterStemmer() #stemming the words to their based form using Porter algorithm

def preprocess(text):
  #remove urls, mentions adn multiple whitespaces

  spacepattern='\s+'

  urlregex=('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|''[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
  mentionregex='@[\w\-]+'

  parsedtext=re.sub(spacepattern, " ", text)
  parsedtext=re.sub(urlregex,"",parsedtext)
  parsedtext=re.sub(mentionregex,"",parsedtext)

  return parsedtext

def tokenize(tweet):
  #We tokenize the weet, remove punctiuation, whitespace, etc and set to small letters,and we stem the tweets

  tweets=" ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
  tokens=[]
  for t in tweets.split():
    tokens.append(stemmer.stem(t))

  return tokens

def tokenizewithoutstemming(tweet):
  tweets=" ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()

  return tweets.split()

#converting data into TFIDF features, that is sying importnace of word in document
'''
parameters:
ngram range: no of contiguous sequence of n words oto be taken from given sample
max features speicifies maxinum number of features or unique words to be extracted fro text, that is words that occur more frequently are given heigher weights
min df and max df specify min and max document frequency that is , control the inclusion of terms that appear too frequently or infrequently
use idf, smoothidf norm are contolling how values are calculated and normalized
'''

vectorizer=TfidfVectorizer(tokenizer=tokenize, preprocessor=preprocess, ngram_range=(1,3), stop_words=stopwords, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=10000, min_df=5, max_df=0.75)

warnings.simplefilter(action='ignore', category=FutureWarning)

#construct tfidf matrix and get scores

tfidf=vectorizer.fit_transform(tweets).toarray() #fitting the vectorizer to ext data and transform to tfidf matrix and then to numpy array
#tfidf contains tfidf matrix row is tweet col is unique word or ngram fe

vocab={}
#map feature names(words or ngrams) to corresponding index in tfidf matrix
featurenames=vectorizer.get_feature_names_out() #get feature names

#enumerate thru feature names and assign index to each feature

for i,v in enumerate(featurenames):
  vocab[v]=i #index i to feature v

idfvals=vectorizer.idf_
idfdict={}
#map feature index to idf score
for featureindex in vocab.values():
  idfdict[featureindex]=idfvals[featureindex]

'''
basically feature names that is words or ngrams are got from vectorize,r each index ssigned feature name, and then vocab contains all feature names as keys, and index correspoinding as values
idfdi t contains mapping of feature index to correspoding idf score'''

# nltk.download('averaged_perceptron_tagger')

#getting part of speech tag
postags=[] #uses pos tags for tweets
for t in tweets:
  tokens=tokenizewithoutstemming(preprocess(t))
  tags=nltk.pos_tag(tokens)
  taglist=[]
  for x in tags:
    taglist.append(x[1])
  tagstring=" ".join(taglist)
  postags.append(tagstring)

#iusing tfidf vecotirzer to get a token matrix for pos tags
#the token matrix is a cloection of text into matrix, where each row represents text and column a unique token ngrma or word extracted fro m thetext data
posvectorizer=TfidfVectorizer(tokenizer=None, lowercase=False, preprocessor=None, ngram_range=(1,3), stop_words=None, use_idf=False, smooth_idf=False, norm=None, decode_error='replace', max_features=5000, min_df=5, max_df=0.75)
#after getting token matrix, the code constructs another matrixrepreosenting pos tags of tweets
#we make another matrix representing pos tags of tweets

pos=posvectorizer.fit_transform(pd.Series(postags)).toarray()

pos_vocab = {v:i for i, v in enumerate(posvectorizer.get_feature_names_out())}

#getting sentiment analysis and other features
sentiment_analyzer=VS()

def countobjects(text):
  '''
  function to count certain media specific objects like url, mentions, hashtags etc
  replaces urls, multiple whitespace, mentions, hasthags, to get standard counts of all these without caring for whos mentioned
  '''

  spacepattern='\s+'
  urlregex=('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|''[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
  mentionregex='@[\w\-]+'
  hashtagregex='#[\w\-]+'
  parsedtext=re.sub(spacepattern, ' ', text)
  parsedtext=re.sub(urlregex, 'URL', parsedtext)
  parsedtext=re.sub(mentionregex,'MENTION', parsedtext)
  parsedtext=re.sub(hashtagregex,'HASHTAG',parsedtext)

  return(parsedtext.count('URL'), parsedtext.count('MENTION'), parsedtext.count('HASHTAG'))

def otherfeatures(tweet):
  #We take stirng, and return list of features like sentiment score, readability score

  sentiment=sentiment_analyzer.polarity_scores(tweet)
  words=preprocess(tweet) #getting only text no charcters
  syllables=textstat.syllable_count(words)
  numchars=sum(len(w) for w in words)
  totalnumchars=len(tweet)
  numwords=len(words.split())
  numterms=len(tweet.split())
  avgsyl=round(float((syllables+0.001))/float(numwords+0.001),4)
  numuniqueterms=len(set(words.split()))


  #Modified FK grade, where avg words per sentence is num words/1
  fkra=round(float(0.39*float(numwords)/1.0)+float(11.8*avgsyl)-15.59,1)
  #fre score
  fre=round(206.835-1.015*(float(numwords)/1.0)-(84.6*float(avgsyl)),2)

  mediaobjects=countobjects(tweet)
  retweet=0
  if "rt" in words:
    retweet=1  #can use this feature of retweeting in forum

  features=[fkra, fre, syllables, avgsyl, numchars, totalnumchars, numterms, numuniqueterms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'], mediaobjects[2], mediaobjects[1], mediaobjects[0], retweet]
  return features

def getfeaturearray(tweets):
  features=[]
  for t in tweets:
    features.append(otherfeatures(t))
  return np.array(features)

otherfeatureslist = ["FKRA", "FRE","num_syllables", "avg_syl_per_word", "num_chars", "num_chars_total", \
                        "num_terms", "num_words", "num_unique_words", "vader neg","vader pos","vader neu", \
                        "vader compound", "num_hashtags", "num_mentions", "num_urls", "is_retweet"]

feat=getfeaturearray(tweets)

#joining everything up
joined=np.concatenate([tfidf, pos, feat], axis=1)

joined.shape

#getting a list of variable names

variables=['']*len(vocab)
for k, v in vocab.items():
  variables[v]=k

posvariables=['']*len(pos_vocab)
for k, v in pos_vocab.items():
  posvariables[v]=k

featurenames=variables+posvariables+otherfeatureslist

x=pd.DataFrame(joined)
y=df['class'].astype(int)

xtrain, xtest, ytrain, ytest=train_test_split(x, y, random_state=42, test_size=0.1)

# pipe=Pipeline([('select', SelectFromModel(LogisticRegression(class_weight='balanced', penalty='l2',C=0.01))), ('model', LogisticRegression(class_weight='balanced', penalty='l2', solver='lbfgs'))])

# paramgrid = {
#     'model__C': [0.01, 0.1, 1.0],  # Regularization parameter values
#     'model__penalty': ['l1', 'l2']  # Type of regularization
# }
# gridsearch=GridSearchCV(pipe, paramgrid, cv=StratifiedKFold(n_splits=5, random_state=None).split(xtrain, ytrain), verbose=2)
# NOTE random state is set to None, if you want 42, set shuffle as True

#NO NEED TO PIPELINING AS WE WANT FAST

model = LogisticRegression(class_weight='balanced', penalty='l2')

model.fit(xtrain, ytrain)

ypred=model.predict(xtest)

report=classification_report(ytest, ypred)

print(report)

confusion_matrix = confusion_matrix(ytest,ypred)
matrix_proportions = np.zeros((3,3))
for i in range(0,3):
    matrix_proportions[i,:] = confusion_matrix[i,:]/float(confusion_matrix[i,:].sum())
names=['Hate','Offensive','Neither']
confusion_df = pd.DataFrame(matrix_proportions, index=names,columns=names)
plt.figure(figsize=(5, 5))
seaborn.heatmap(confusion_df,annot=True,annot_kws={"size": 12},cmap='gist_gray_r',cbar=False, square=True,fmt='.2f')
plt.ylabel(r'True categories',fontsize=14)
plt.xlabel(r'Predicted categories',fontsize=14)
plt.tick_params(labelsize=12)

from google.colab import drive
drive.mount('/content/drive')

# import joblib
# model_path = '/content/drive/My Drive/Colab Notebooks/hatespeechmodel.pkl'
# joblib.dump(model, model_path)

# print(ypred[69])

