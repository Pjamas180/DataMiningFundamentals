import math
import random
import itertools
import gzip
import time
import numpy
from collections import defaultdict
from datetime import datetime

startTime = datetime.now()
print startTime

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

user_list = set()
item_list = set()

itemCount = defaultdict(int)
userPurchaseCount = defaultdict(int)

I_u = defaultdict(list)
U_i = defaultdict(list)

totalPurchases = 0
for l in readGz("train.json.gz"):
  user,item,rating,category = l['reviewerID'],l['itemID'],l['rating'],l['category']
  user_list.add(user)
  item_list.add(item)
  I_u[user].append({'itemID': item, 'rating': rating, 'category': category})
  U_i[item].append({'reviewerID': user, 'rating': rating, 'category': category})
  itemCount[item] += 1
  userPurchaseCount[user] += 1

  totalPurchases += 1
print "1 - ", totalPurchases, "\n"

mostPopular = [(itemCount[x], x) for x in itemCount]
mostPopular.sort()
mostPopular.reverse()

mostActive = [(userPurchaseCount[x], x) for x in userPurchaseCount]
mostActive.sort()
mostActive.reverse()

return1 = set()
count2 = 0

for ic, i in mostPopular:
  count2 += ic
  return1.add(i)
  if count2 > totalPurchases/3: break

return2 = set()
count = 0

# Modification 2 - consider how active a user is
threshold = totalPurchases/15 # 0.70136

for upc, u in mostActive:
  count += upc
  return2.add(u)
  if count > threshold: break

print "2\n"
randomGuessCount = []

def makePrediction(user, item):
  if user not in user_list:
    # If we don't know about the user, then see how popular it is?
    randomGuessCount.append((user,item))
    return 1;
  if item not in item_list: 
    # If we don't know about the item, then see how active user is?
    randomGuessCount.append((user,item))
    return 1;
  otherItemsBoughtByUser = [entry['itemID'] for entry in I_u[user]]

  # Finally, If this item is similar enough to one of the other items this 
  # user bought, predict 1
  for otherItem in otherItemsBoughtByUser:
    similarity = PearsonItemSimilarity(i, otherItem)
    if math.fabs(similarity) > 0.5: 
      #print similarity
      return 1
  if item in return1 or user in return2:
    return 1
  return 0

def PearsonItemSimilarity(item1, item2):
  AvgRatingForItem1 = (sum([float(u['rating']) for u in U_i[item1]]) * 1.0) / len( [u for u in U_i[item1]] )
  AvgRatingForItem2 = (sum([float(u['rating']) for u in U_i[item2]]) * 1.0) / len( [u for u in U_i[item2]] )
  U_i1 = set( [user['reviewerID'] for user in U_i[item1]] )
  U_i2 = set( [user['reviewerID'] for user in U_i[item2]] )
  UsersThatReviewedBoth = U_i1.intersection(U_i2)
  numerator = sum( [(rating(u,item1) - AvgRatingForItem1)*(rating(u,item2) - AvgRatingForItem2) for u in UsersThatReviewedBoth] )
  denominator = math.sqrt( sum( [(rating(u,item1) - AvgRatingForItem1)**2 for u in UsersThatReviewedBoth] ) * sum( [(rating(u,item2) - AvgRatingForItem2)**2 for u in UsersThatReviewedBoth] ) )
  similarity = 0.0 if denominator == 0.0 else numerator/denominator
  return similarity

def rating(user, item):
  for i in I_u[user]:
    if i['itemID'] == item: return float(i['rating'])

pairs = "pairs_Purchase.txt"
predictions = open("predictions_Purchase.txt", 'w')
c = 0
for l in open(pairs):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i = l.strip().split('-')
  c += 1
  #print c
  predictions.write(u + '-' + i + "," + str(makePrediction(u, i)) + "\n")

predictions.close()

predictions = "predictions_Purchase.txt"
cnt = 0
numOnes = 0
for l in open(predictions):
  if l.startswith("userID"):
    #header
    continue
  x,r = l.strip().split(',')
  cnt += 1
  if int(r) == 1: numOnes += 1

print "Made ", len(randomGuessCount), " random guesses\n"
print float(numOnes)/float(cnt), "\n"

print datetime.now() - startTime
