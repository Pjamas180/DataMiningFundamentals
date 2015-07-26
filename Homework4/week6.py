import numpy
import urllib
import scipy.optimize
import random
from collections import defaultdict
import nltk
import string
from nltk.stem.porter import *
from sklearn import linear_model
import operator
import math
from textblob import TextBlob as tb

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

### Just the first 5000 reviews

print "Reading data..."
data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))[:5000]
print "done"

### How many unique words are there?

count = 0
wordCount = defaultdict(int)
for d in data:
  if count > 1:
    break
  count += 1
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  for w in r.split():
    wordCount[w] += 1


print len(wordCount)

### Ignore capitalization and remove punctuation

wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  kWords = r.split();
  for index,w in enumerate(kWords):
    if index < len(kWords)-1:
      word = w + " " + kWords[index+1]
      wordCount[word] += 1

# Number of bigrams
#####################~~~ TASK 1 ~~~#####################
bigramCount = len(wordCount)

wordCount1 = sorted(wordCount.items(), key=operator.itemgetter(1))
wordCount1.reverse()

topFive = wordCount1[:5]

#####################~~~ TASK 2 ~~~#####################

# Top 1000 bigrams
top1000 = [x[0] for x in wordCount1[:1000]]

# Create a feature for bigrams
def feature2(datum):
  feat = [0]*len(top1000)
  r = ''.join([c for c in datum['review/text'].lower() if not c in punctuation])
  kWords = r.split()
  for index, w in enumerate(kWords):
    if index < len(kWords)-1:
      word = w + " " + kWords[index+1] ##bigrams
    if word in top1000:
      feat[wordId[word]] += 1
  feat.append(1) #offset
  return feat

wordId = dict(zip(top1000, range(len(top1000))))
X = [feature2(d) for d in data]
y = [d['review/overall'] for d in data]
theta2,residuals,rank,s = numpy.linalg.lstsq(X, y)

#####################~~~ TASK 3 ~~~#####################

def featureUB(datum):
  feat = [0]*len(words)
  r = ''.join([c for c in datum['review/text'].lower() if not c in punctuation])
  kWords = r.split()
  for index, w in enumerate(kWords):
    if index < len(kWords)-1:
      word = w + " " + kWords[index+1] ##bigrams
      if word in top1000a:
        feat[wordId[word]] += 1
      if w in top1000a:
        feat[wordId[w]] += 1
  feat.append(1) #offset
  return feat

wordCount2 = defaultdict(int)
punctuation = set(string.punctuation)
for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  for w in r.split():
    wordCount2[w] += 1

counts = sorted(wordCount2.items(), key=operator.itemgetter(1))
counts.reverse()

uniAndBigrams = wordCount1 + counts
uniAndBigrams = sorted(uniAndBigrams, key=operator.itemgetter(1))
uniAndBigrams.reverse()

top1000a = [x[0] for x in uniAndBigrams[:1000]]

wordId = dict(zip(top1000a, range(len(top1000a))))
X = [featureUB(d) for d in data]
y = [d['review/overall'] for d in data]

theta3,residuals,rank,s = numpy.linalg.lstsq(X, y)

#####################~~~ TASK 4 ~~~#####################

theta3.sort()
maxValues = theta3[len(theta3)-5:]
minValues = theta3[5:]

zipped = zip(theta3,top1000a)
minItems = zipped[:5]
maxItems = zipped[len(zipped)-5:]

#####################~~~ TASK 5 ~~~#####################

# unigram predictor values
theta.sort()
zipped1 = zip(theta,words)
minItems1 = zipped1[:5]
maxItems1 = zipped1[len(zipped1)-5:]


# bigram predictor values
theta2.sort()
zipped2 = zip(theta2,top1000)
minItems2 = zipped2[:5]
maxItems2 = zipped2[len(zipped2)-5:]

# Both predictor values above in task 4

#####################~~~ TASK 6 ~~~#####################

def tf(word, blob):
  return float(float(blob.words.count(word)) / float(len(blob.words)))

def n_containing(word, bloblist):
  return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
  return float(math.log10(len(bloblist) / float((1 + n_containing(word, bloblist)))))

def tfidf(word, blob, bloblist):
  return float(tf(word, blob) * idf(word, bloblist))

# Get list of all reviews in blobs
disBlobList = []
for d in data:
  review = d['review/text']
  disBlobList.append(tb(review))

# The higher the idf, the less frequent the word appears throughout all documents - could be important.
# The lower the tfidf, the less important the word is to the document.

# 'foam'
foam = idf('foam',disBlobList)
foam2 = tfidf('foam', disBlobList[0], disBlobList)
# 'smell'
smell = idf('smell',disBlobList)
smell2 = tfidf('smell', disBlobList[0], disBlobList)
# 'banana'
banana = idf('banana',disBlobList)
banana2 = tfidf('banana',disBlobList[0], disBlobList)
# 'lactic'
lactic = idf('lactic',disBlobList)
lactic2 = tfidf('lactic', disBlobList[0], disBlobList)
# 'tart'
tart = idf('tart',disBlobList)
tart2 = tfidf('tart',disBlobList[0], disBlobList)

#####################~~~ TASK 7 ~~~#####################

dbl0 = ''.join([c for c in disBlobList[0].lower() if not c in punctuation])
dbl1 =''.join([c for c in disBlobList[1].lower() if not c in punctuation])

dbla = set(dbl0.split())
dblb = set(dbl1.split())

newset = set(dbla).intersection(dblb)

dict1 = defaultdict(int)
dict2 = defaultdict(int)

for w in dbl0.split():
  if w in newset:
    dict1[w] += 1

for w in dbl1.split():
  if w in newset:
    dict2[w]+=1

tfidf1 = []
tfidf2 = []
for key in newset:
  tfidf1.append(tfidf(key,disBlobList[0],disBlobList))
  tfidf2.append(tfidf(key,disBlobList[1],disBlobList))

# Calculate the Cosine Similarity
cosineSimilarity = 1 - scipy.spatial.distance.cosine(tfidf1,tfidf2)

#####################~~~ TASK 8 ~~~#####################
cosSimilarity = []
for x in range(1,len(disBlobList)):
  dbl = ''.join([c for c in disBlobList[x].lower() if not c in punctuation])
  dbl7 = set(dbl.split())
  newset = set(dbl7).intersection(dbla)
  tfidf1 = []
  tfidf2 = []
  for key in newset:
    tfidf1.append(tfidf(key,disBlobList[0],disBlobList))
    tfidf2.append(tfidf(key,disBlobList[x],disBlobList))
  cosineSimilarity = 1 - scipy.spatial.distance.cosine(tfidf1,tfidf2)
  cosSimilarity.append((cosineSimilarity, x))

cosSimilarity.max()

### With stemming

wordCount = defaultdict(int)
punctuation = set(string.punctuation)
stemmer = PorterStemmer()
for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  for w in r.split():
    w = stemmer.stem(w)
    wordCount[w] += 1

### Just take the most popular words...

wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  for w in r.split():
    wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

words = [x[1] for x in counts[:1000]]



### Sentiment analysis

wordId = dict(zip(words, range(len(words))))
wordSet = set(words)

def feature(datum):
  feat = [0]*len(words)
  r = ''.join([c for c in datum['review/text'].lower() if not c in punctuation])
  for w in r.split():
    if w in words:
      feat[wordId[w]] += 1
  feat.append(1) #offset
  return feat

X = [feature(d) for d in data]
y = [d['review/overall'] for d in data]

#No regularization
#theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

#With regularization
clf = linear_model.Ridge(1.0, fit_intercept=False)
clf.fit(X, y)
theta = clf.coef_
predictions = clf.predict(X)
