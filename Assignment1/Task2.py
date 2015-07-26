# Helpfulness Prediction

"""
Predict whether a user's review of an item will be considered helpful. The file 'pairs_Helpful.txt' contains (user,item)
pairs, with a third columns containing the number of votes the user's review of the item received, you must predict how
many of them were helpful. 
"""

import gzip
from collections import defaultdict
import math
import numpy
import urllib
import scipy.optimize
import random
import json

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

# Using the logistic regressors
def feature(datum):
  feat = [1]
  feat.append(datum)
  return feat
### Helpfulness baseline: similar to the above. Compute the global average helpfulness rate, and the average helpfulness rate for each user

allHelpful = []
userHelpful = defaultdict(list)
count = 0
firstHelpful = []
lastHelpful = []
allRatings = []
# middleHelpful = []

for l in readGz("train.json.gz"):
  user,item,helpfulRate = l['reviewerID'],l['itemID'],l['helpful']
  allHelpful.append(l['helpful'])
  userHelpful[user].append(l['helpful'])
  allRatings.append(l['rating'])



X = [feature(d) for d in allRatings]
y = [d['nHelpful']/d['outOf'] for d in allHelpful]

theta,residuals,rank,s = numpy.linalg.lstsq(X,y)

allHelpful.sort()

for x in allHelpful:
  count += 1
  if count > 900000:
    lastHelpful.append(x)
  else:
    firstHelpful.append(x)
    """
  else:
    middleHelpful.append(x)
    """

# Takes the average of the Helpful rates for all reviews in the train.
"""
We need to change the way we calculate when a user doesn't show up in the training set
"""
averageRate = sum([x['nHelpful'] for x in allHelpful]) * 1.0 / sum([x['outOf'] for x in allHelpful[500000:]])
firstAverage = sum([x['nHelpful'] for x in firstHelpful]) * 1.0 / sum([x['outOf'] for x in firstHelpful])
lastAverage = sum([x['nHelpful'] for x in lastHelpful]) * 1.0 / sum([x['outOf'] for x in lastHelpful])

# middleAverage = sum([x['nHelpful'] for x in middleHelpful]) * 1.0 / sum([x['outOf'] for x in middleHelpful])

# That user's average helpful rate
userRate = {}
for u in userHelpful:
  userRate[u] = sum([x['nHelpful'] for x in userHelpful[u]]) * 1.0 / sum([x['outOf'] for x in userHelpful[u]])

predictions = open("predictions_Helpful1.txt", 'w')
for l in open("pairs_Helpful.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i,outOf = l.strip().split('-')
  outOf = int(outOf)
  if u in userRate and outOf < 1:
    newRate = (outOf*userRate[u] + (outOf*lastAverage)*4)/5
    predictions.write(u + '-' + i + '-' + str(outOf) + ',' + str(newRate) + '\n')
  else:
      predictions.write(u + '-' + i + '-' + str(outOf) + ',' + str(outOf*averageRate) + '\n')

predictions.close()


"""
      if outOf < 3:
        predictions.write(u + '-' + i + '-' + str(outOf) + ',' + str(outOf*firstAverage) + '\n')
      else:
        predictions.write(u + '-' + i + '-' + str(outOf) + ',' + str(outOf*lastAverage) + '\n')
"""
