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

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

def feature(datum):
  feat = [0]*len(words)
  r = ''.join([c for c in datum['review/text'].lower() if not c in punctuation])
  for w in r.split():
    if w in words:
      feat[wordId[w]] += 1
  feat.append(1) #offset
  return feat
  
def featureB(datum):
  feat = [0]*len(words)
  r = ''.join([c for c in datum['review/text'].lower() if not c in punctuation])
  kWords = r.split()
  for index, w in enumerate(kWords):
	if index < len(kWords)-1:
		word = w + " " + kWords[index+1] ##bigrams
		if word in words:
			feat[wordId[word]] += 1
  feat.append(1) #offset
  return feat

def featureUB(datum):
  feat = [0]*len(words)
  r = ''.join([c for c in datum['review/text'].lower() if not c in punctuation])
  kWords = r.split()
  for index, w in enumerate(kWords):
	if index < len(kWords)-1:
		word = w + " " + kWords[index+1] ##bigrams
		if word in ubWords:
			feat[wordId[word]] += 1
		if w in ubWords:
			feat[wordId[w]] += 1
  feat.append(1) #offset
  return feat

### Just the first 5000 reviews

print "Reading data..."
data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))[:5000]
print "done"

### Ignore capitalization and remove punctuation

wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  kWords = r.split()
  for index, w in enumerate(kWords):
	if index < len(kWords)-1:
		word = w + " " + kWords[index+1] ##bigrams
		wordCount[word] += 1
		
uniqueBigramCount = len(wordCount)
## Number of Unique Bigrams
print "There are " + str(uniqueBigramCount) + " unique bigrams.\n"

sWordCount = sorted(wordCount.items(), key=operator.itemgetter(1))
sWordCount.reverse()

#Top Five Bigrams
print "The top five bigrams are as follows:"
for i in sWordCount[:5]:
	print i
print "\n"
	
#Get the top 1000 most popular bigrams
words = [x[0] for x in sWordCount[:1000]]

wordId = dict(zip(words, range(len(words))))
X = [featureB(d) for d in data]
y = [d['review/overall'] for d in data]

#No regularization
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

#Bigrams residual
print "The residuals obtained using the new predictor for bigrams is " + str(residuals[0]) + "\n"

wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  for w in r.split():
    wordCount[w] += 1

counts = sorted(wordCount.items(), key=operator.itemgetter(1))
counts.reverse()

#Combine unigrams and bigrams
uniBigrams = sWordCount + counts
uniBigrams = sorted(uniBigrams, key=operator.itemgetter(1))
uniBigrams.reverse()

#Get the top 1000 most popular unigrams and bigrams
ubWords = [x[0] for x in uniBigrams[:1000]]

wordId = dict(zip(ubWords, range(len(ubWords))))
X = [featureUB(d) for d in data]
y = [d['review/overall'] for d in data]

#No regularization
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

#Bigrams residual
print "The residuals obtained using the new predictor for unigrams and bigrams is " + str(residuals[0]) + "\n"

#The most positive weighted unigrams/bigrams
theta

#Normal unigrams provided
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

wordId = dict(zip(words, range(len(words))))
X = [feature(d) for d in data]
y = [d['review/overall'] for d in data]

#No regularization
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

#Unigrams residual
print "The residuals obtained using the new predictor for unigrams is " + str(residuals[0]) + "\n"

#is idf = log (number of documents / number of times phrase appears in all documents) ?




# c = 0
# for w in sWordCount:
	# print k
	# c += 1
	# if c > 10:
		# break

# print len(wordCount)

# ### With stemming

# # wordCount = defaultdict(int)
# # punctuation = set(string.punctuation)
# # stemmer = PorterStemmer()
# # for d in data:
  # # r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  # # for w in r.split():
    # # w = stemmer.stem(w)
    # # wordCount[w] += 1

# ### Just take the most popular words...

# wordCount = defaultdict(int)
# punctuation = set(string.punctuation)
# for d in data:
  # r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  # for w in r.split():
    # wordCount[w] += 1

# counts = sorted(wordCount.items(), key=operator.itemgetter(1))
# counts.reverse()

# #Combine unigrams and bigrams
# uniBigrams = sWordCount + counts
# uniBigrams = sorted(uniBigrams, key=operator.itemgetter(1))
# uniBigrams.reverse()

# #Get the top 1000 most popular unigrams and bigrams
# ubWords = [x[0] for x in uniBigrams[:1000]]

# ### Sentiment analysis

# wordId = dict(zip(words, range(len(words))))
# wordSet = set(words)

# def feature(datum):
  # feat = [0]*len(words)
  # r = ''.join([c for c in datum['review/text'].lower() if not c in punctuation])
  # for w in r.split():
    # if w in words:
      # feat[wordId[w]] += 1
  # feat.append(1) #offset
  # return feat

# X = [feature(d) for d in data]
# y = [d['review/overall'] for d in data]

# #No regularization
# theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

# #With regularization
# # clf = linear_model.Ridge(1.0, fit_intercept=False)
# # clf.fit(X, y)
# # theta = clf.coef_
# # predictions = clf.predict(X)
# # score = clf.score(X, y)
