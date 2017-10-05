### Problem 1 Solution
import numpy
import urllib
import scipy.optimize
import random

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))
print "done"

print "### Week 1 Part 1 ###"

def unique(identifier, datah):
	return set([x[identifier] for x in datah])

### Get the Unique number of items according to 'beer/beerId'
uniqueBeerId = unique('beer/beerId', data)
print "Unique number of items according to 'beer/beerId': " + str(len(uniqueBeerId))

#uniqueUsers = set([x['user/profileName'] for x in data])
uniqueUsers = unique('user/profileName', data)
print "Unique number of users according to 'user/profileName': " + str(len(uniqueUsers))

### Calculating the mean for each of the 5 ratings: 'review/appearance', 'review/palate', 'review/overall', 
### 'review/aroma', 'review/taste'.

def mean(identifier, datah):
	l = [x[identifier] for x in datah]
	return sum(l)/float(len(l))

ratingAppearanceAvg = mean('review/appearance', data)
print "Mean of 'review/appearance': " + str(ratingAppearanceAvg)

ratingPalateAvg = mean('review/palate', data)
print "Mean of 'review/appearance': " + str(ratingPalateAvg)

ratingOverallAvg = mean('review/overall', data)
print "Mean of 'review/overall': " + str(ratingOverallAvg)

ratingAromaAvg = mean('review/aroma', data)
print "Mean of 'review/aroma': " + str(ratingAromaAvg)

ratingTasteAvg = mean('review/taste', data)
print "Mean of 'review/taste': " + str(ratingTasteAvg)

### Calculating mean ABV

avgABV = mean('beer/ABV', data)
print "Mean of 'beer/ABV': " + str(avgABV)


print "### Week 1 Part 2 ###"

### Calculating variance of 'review/taste'.

varTaste = numpy.var([x['review/taste'] for x in data])
print "Variance of 'review/taste': " + str(varTaste)

print "Since the mean of the taste is all we have as a prediction, the MSE is the same"
print "as the variance."

print "MSE of 'review/taste': " + str(varTaste)

### Predicting the taste rating ('review/taste') as a function of the ABV ('beer/ABV') of a certain beer.

print "### Week 1 Part 3 ###"

data2 = [d for d in data if d.has_key('beer/ABV')]

def feature(datum):
  feat = [1]
  feat.append(datum['beer/ABV'])
  return feat

X = [feature(d) for d in data2]
y = [d['review/taste'] for d in data2]
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

print "Fitted values for theta[0]: " + str(theta[0]) + " and theta[1]: " + str(theta[1])

print "Theta[0] means that at an ABV of zero, the taste will probably be at around a 3."

print "As a function of a beer's ABV, the taste increases by " + str(theta[1])

### Dividing the dataset into two where the first half is the training set and the second half for test.

print "### Week 1 Part 4###"
data1 = data[:25000]
data2 = data[25000:50000]

trainingSet = [d for d in data1 if d.has_key('beer/ABV')]

X1 = [feature(d) for d in trainingSet]
y1 = [d['review/taste'] for d in trainingSet]
theta,residuals,rank,s = numpy.linalg.lstsq(X1, y1)

print "Predictor obtained from training set: " + str(theta[1])

### Calculating the MSE on the training set after finding the variance
### varTraining = numpy.var([x['review']])

blahh = [(theta[0] + (theta[1]*d['beer/ABV'])) for d in trainingSet]
zipped = [a - b for a, b in zip(y1, blahh)]
squared = [a**2 for a in zipped]
blah = sum(squared)
mse = blah / 25000

print "Mean-squared error: " + str(mse)

### avgTaste = mean('review/taste', data1)
### mse = ((theta[1] - avgTaste) ** 2).mean(axis=None)

testSet = [d for d in data2 if d.has_key('beer/ABV')]

X2 = [feature(d) for d in testSet]
y2 = [d['review/taste'] for d in testSet]
# theta,residuals,rank,s = numpy.linalg.lstsq(X1, y1)

blahh = [(theta[0] + (theta[1]*d['beer/ABV'])) for d in testSet]
zipped = [a - b for a, b in zip(y2, blahh)]
squared = [a**2 for a in zipped]
blah = sum(squared)
mse2 = blah / 25000

### Now we are going to incorporate the time of day where we set a certain hour according
### to a binary system and see how that affects the taste. 'review/timeStruct'

data3 = [d for d in data if d.has_key('review/timeStruct')]

def feature(datum):
  feat = [1]
  if datum['review/timeStruct'].get('hour') == 0:
    feat.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
  elif datum['review/timeStruct'].get('hour') == 1:
    feat.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
  elif datum['review/timeStruct'].get('hour') == 2:
  	feat.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
  elif datum['review/timeStruct'].get('hour') == 3:
  	feat.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])
  elif datum['review/timeStruct'].get('hour') == 4:
  	feat.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
  elif datum['review/timeStruct'].get('hour') == 5:
  	feat.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
  elif datum['review/timeStruct'].get('hour') == 6:
  	feat.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
  elif datum['review/timeStruct'].get('hour') == 7:
  	feat.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
  elif datum['review/timeStruct'].get('hour') == 8:
  	feat.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
  elif datum['review/timeStruct'].get('hour') == 9:
  	feat.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
  elif datum['review/timeStruct'].get('hour') == 10:
  	feat.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
  elif datum['review/timeStruct'].get('hour') == 11:
  	feat.extend([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
  elif datum['review/timeStruct'].get('hour') == 12:
  	feat.extend([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
  elif datum['review/timeStruct'].get('hour') == 13:
  	feat.extend([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
  elif datum['review/timeStruct'].get('hour') == 14:
  	feat.extend([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
  elif datum['review/timeStruct'].get('hour') == 15:
  	feat.extend([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
  elif datum['review/timeStruct'].get('hour') == 16:
  	feat.extend([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
  elif datum['review/timeStruct'].get('hour') == 17:
  	feat.extend([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
  elif datum['review/timeStruct'].get('hour') == 18:
  	feat.extend([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
  elif datum['review/timeStruct'].get('hour') == 19:
  	feat.extend([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
  elif datum['review/timeStruct'].get('hour') == 20:
  	feat.extend([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
  elif datum['review/timeStruct'].get('hour') == 21:
  	feat.extend([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
  elif datum['review/timeStruct'].get('hour') == 22:
  	feat.extend([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
  elif datum['review/timeStruct'].get('hour') == 23:
  	feat.extend([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
  return feat


data1 = data3[:25000]
data2 = data3[25000:50000]

trainingSet = [d for d in data1 if d.has_key('review/timeStruct')]

X1 = [feature(d) for d in trainingSet]
y1 = [d['review/taste'] for d in trainingSet]
theta,residuals,rank,s = numpy.linalg.lstsq(X1, y1)

# print "Predictor obtained from training set: " + str(theta[1])

### Calculating the MSE on the training set after finding the variance
### varTraining = numpy.var([x['review']])

blahh = [(theta[0] + (theta[1]*d['beer/ABV'])) for d in trainingSet]
zipped = [a - b for a, b in zip(y1, blahh)]
squared = [a**2 for a in zipped]
blah = sum(squared)
mse = blah / 25000

print "Mean-squared error on training set: " + str(mse)

testSet = [d for d in data2 if d.has_key('review/timeStruct')]

X2 = [feature(d) for d in testSet]
y2 = [d['review/taste'] for d in testSet]
theta,residuals,rank,s = numpy.linalg.lstsq(X2,y2)

blahh = [(theta[0] + (theta[1]*d['beer/ABV'])) for d in testSet]
zipped = [a - b for a, b in zip(y1, blahh)]
squared = [a**2 for a in zipped]
blah = sum(squared)
mse2 = blah / 25000

print "Mean-squared error on training set: " + str(mse2)

