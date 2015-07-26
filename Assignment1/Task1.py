# Purchase Prediction

import gzip
from collections import defaultdict
import math
import numpy
import urllib
import scipy.optimize
import random
import json
from sklearn.metrics import accuracy_score

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)



def diff_gender_clothing(user, item):
  clothing = 'Clothing, Shoes & Jewelry'
  if user not in I_u or item not in U_i:
    return False 
  item_cw = category_words(U_i[item][0]['category'])
  if clothing in item_cw:
    for i in I_u[user]:
      i_cw = category_words(i['category'])
      if clothing not in i_cw:
        continue
      if ('Men' in i_cw and 'Women' in item_cw) or ('Women' in i_cw and 'Men' in item_cw):
        return True
  return False

def category_words(category):
  return set([word for subcategory in category for word in subcategory])

def pearson_item(i1, i2):
  R_i1_avg = numpy.mean([u['rating'] for u in U_i[i1]])
  R_i2_avg = numpy.mean([u['rating'] for u in U_i[i2]])
  R_intersection = [{'i1': u['rating'], 'i2': v['rating']} for u in U_i[i1] for v in U_i[i2] if u['reviewerID'] == v['reviewerID']]
  numerator = (sum([(r['i1'] - R_i1_avg) * (r['i2'] - R_i2_avg) for r in R_intersection])) * 1.0
  denominator = (sum([(r['i1'] - R_i1_avg) ** 2 for r in R_intersection]) * sum([(r['i2'] - R_i2_avg) ** 2 for r in R_intersection])) ** 0.5
  return 0.0 if denominator == 0.0 else numerator / denominator 

def pearson(user, item):
  if user not in I_u or item not in U_i:
    return True 
  for i in I_u[user]:
    if pearson_item(i['itemID'], item) > 0.5:
      return True
  return False

# Classification Accuracy

### Purchasing baseline: just rank which items are popular and which are not, and return '1' if an item is among the top-ranked

itemCount = defaultdict(int)
userCount = defaultdict(int)
I_u = {k:[] for k in range(794768)}
U_i = {k:[] for k in range(244412)}
totalPurchases = 0


for l in readGz("train.json.gz"):
  user,item,category = l['reviewerID'],l['itemID'],l['category']
  itemCount[item] += 1
  totalPurchases += 1
  # Figure out how active a user is.
  userCount[user] += 1
  if item not in U_i:
    U_i[item] = [l]
  U_i[item].append(l)
  if user not in I_u:
    I_u[user] = [l]
  I_u[user].append(l)


mostPopular = [(itemCount[x], x) for x in itemCount]
mostPopular.sort()
mostPopular.reverse()

mostActive = [(userCount[x],x) for x in userCount]
mostActive.sort()
mostActive.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
  count += ic
  return1.add(i)
  if count > (totalPurchases*2)/3: break

return2 = set()
active = 0
for ic, i in mostActive:
  active += ic
  return2.add(i)
  if active > (totalPurchases*1)/12: break

"""
predictions = open("predictions_Purchase.txt", 'w')
for l in open("pairs_Purchase.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i = l.strip().split('-')
  if i in return1 or u in return2:
    if diff_gender_clothing(u, i):
      predictions.write(u + '-' + i + ",0\n")
    else:
      if pearson(u, i):
        predictions.write(u + '-' + i + ",1\n")
      else:
        predictions.write(u + '-' + i + ",0\n")
  else:
    predictions.write(u + '-' + i + ",0\n")

predictions.close()
"""

predictions = open("predictions_Purchase.txt", 'w')
for l in open("pairs_Purchase.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i = l.strip().split('-')
  if pearson(u,i):
    if diff_gender_clothing(u, i):
      predictions.write(u + '-' + i + ",0\n")
    else:
      if i in return1 or u in return2:
        predictions.write(u + '-' + i + ",1\n")
      else:
        predictions.write(u + '-' + i + ",0\n")
  else:
    predictions.write(u + '-' + i + ",0\n")

predictions.close()



