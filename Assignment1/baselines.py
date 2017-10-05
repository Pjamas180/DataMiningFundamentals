###---------------------- DEFINE DEPENDENCIES 
import gzip
import urllib
import collections
from collections import defaultdict
import math
import numpy
import re
from string import punctuation

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)
	
def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)
	
#Find common elements
def jac(u1,u2, datam):
	common = set(datam[u1]) & set(datam[u2])
	numCommon = len(common)
	union = set(datam[u1]) | set(datam[u2])
	numUnion = len(union)
	#JACCARD
	jacc = float(numCommon) / float(numUnion)
	return jacc

#BASELINES
itemCount = defaultdict(int)
totalPurchases = 0

I_u = {};
U_i = {};

for l in readGz("train.json.gz"):
  user,item = l['reviewerID'],l['itemID']
  itemCount[item] += 1
  totalPurchases += 1
  if (user not in I_u): 
    I_u.update({user: [l]})
  else:
    I_u[user].append(l)
  if (item not in U_i):
    U_i.update({item: [l]})
  else:
    U_i[item].append(l)


#Pearson Algorithm
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
    if math.fabs(pearson_item(i['itemID'], item)) > 0.5:
      return True
  return False
  
def tree():
    return collections.defaultdict(tree)

###---------------------- DEFINE DEPENDENCIES 

###---------------------- EXTRACT DATA

#Extract the Data
data = list(parseData("train.json"))
#myData = data[:len(data)/2]
#testData = data[len(data)/2:]

###---------------------- EXTRACT DATA

### Helpfulness baseline: similar to the above. Compute the global average helpfulness rate, and the average helpfulness rate for each user

allHelpful = []
userHelpful = defaultdict(list)
ratingHelpful = defaultdict(list)
wcHelpful =  defaultdict(list)

for l in readGz("train.json.gz"):
  user,item,rating,reviewText = l['reviewerID'],l['itemID'],l['rating'],l['reviewText']
  wc = len(re.split(r'[^0-9A-Za-z]+',reviewText))
  
  allHelpful.append(l['helpful'])
  userHelpful[user].append(l['helpful'])
  ratingHelpful[rating].append(l['helpful'])
  wcHelpful[wc].append(l['helpful'])

predData = tree()
for l in readGz("helpful.json.gz"):
	user,item,rating,reviewText = l['reviewerID'],l['itemID'],l['rating'],l['reviewText']
	
	wc = len(re.split(r'[^0-9A-Za-z]+',reviewText))
	predData[user][item]['rating'] = rating
	predData[user][item]['wordCount'] = wc
 
allHelpful.sort()

#Average Rate
averageRate = sum([x['nHelpful'] for x in allHelpful]) * 1.0 / sum([x['outOf'] for x in allHelpful])

#User Rate difference
#userRate = {}
#for u in userHelpful:
#  userRate[u] =  (sum([x['nHelpful'] for x in userHelpful[u]]) * 1.0 / sum([x['outOf'] for x in userHelpful[u]]))

#Rat Rate difference
ratRate = {}
for u in ratingHelpful:
  ratRate[u] =  (sum([x['nHelpful'] for x in ratingHelpful[u]]) * 1.0 / sum([x['outOf'] for x in ratingHelpful[u]])) - averageRate

#WC Rate difference
wcRate = {}
for u in wcHelpful:
  wcRate[u] =  (sum([x['nHelpful'] for x in wcHelpful[u]]) * 1.0 / sum([x['outOf'] for x in wcHelpful[u]])) - averageRate

  
  
wut = 0
#Total helpfuls difference
totalRate = defaultdict(list)
for a in allHelpful:
	if a['outOf'] in totalRate:
		totalRate[a['outOf']].append(a['nHelpful'])
	else:
		if a['nHelpful'] != []:
			totalRate[a['outOf']] = [a['nHelpful']]
for a in totalRate:
	totalRate[a] = (numpy.mean(totalRate[a])/a) - averageRate
	
		
  
predictions = open("predictions_Helpful.txt", 'w')
for l in open("pairs_Helpful.txt"):
	flag = 0
	if l.startswith("userID"):
	#header
		predictions.write(l)
		continue
	u,i,outOf = l.strip().split('-')
	outOf = int(outOf)
	rating = predData[u][i]['rating']
	wc = predData[u][i]['wordCount']
	
	total = outOf

	totalH = 0
	rate = averageRate
	#if u in userRate:
	#	rate = userRate[u]
	#if rate < 1:
	while outOf not in totalRate:
		outOf -= 1
		if outOf == 0:
			break
	if outOf in totalRate:
		rate += totalRate[outOf]
	
	while rating not in totalRate:
		rating -= 1
		if rating == 0:
			break
	if rating in ratRate:
		rate += ratRate[rating]
		
	while wc not in totalRate:
		wc -= 1
		if wc == 0:
			break
	if wc in wcRate:
		rate += wcRate[wc]
	
	if rate > 1:
		rate = 1
	totalH = total * rate
	
	totalH = int(round(totalH))

	predictions.write(u + '-' + i + '-' + str(total) + ',' + str(totalH) + '\n')

predictions.close()

### Purchasing baseline: just rank which items are popular and which are not, and return '1' if an item is among the top-ranked

#Extract List of Data
userItem = [(l['reviewerID'],l['itemID'],l['rating']) for l in data]

##START -- Create Dictionaries
dic = defaultdict(list)
dic2 = defaultdict(list)

for k, v in userItem:
	#Create Users = [Item1, Item2, Item3]
	dic[k].append(v)
	#Create Items = [User1, User2, User3]
	dic2[v].append(k)
	
userItem = dic
itemUser = dic2
##END -- Create Dictionaries

items = [(l['itemID'],l['category']) for l in data]
dic3 = defaultdict(list)
for k, v in items:
	dic3[k].append(v)
items = dic3

iM = 0
iF = 0
iG = 0
iB = 0
itemCat = defaultdict(list)
itemCat2 = defaultdict(list)
#Categorize Items as M or F or G or B
for i in items:
	for m in items[i]:
		itemCat[i] = 'N'
		for s in m:
			if 'Men' in s:
				itemCat[i] = 'M'
				iM += 1
			elif 'Women' in s:
				itemCat[i] = 'F'
				iF += 1
			elif 'Girl' in s:
				itemCat[i] = 'G'
				iG += 1
			elif 'Boy' in s:
				itemCat[i] = 'B'
				iB += 1

userGender = defaultdict(list)
userCats = defaultdict(list)
M = 0
F = 0
N = 0
G = 0
B = 0

ath = 0

#Label Users as M or F or G or B
for u in userItem:
	genderSet = 0
	gender = 'N'
	for it in userItem[u]:
		#Figure out gender
		if itemCat[it] == 'M':
			if genderSet == 1 and gender != 'M':
				gender = 'N'
				break
			else:
				gender = 'M'
				genderSet = 1
		elif itemCat[it] == 'F':
			if genderSet == 1 and gender != 'F':
				gender = 'N'
				break
			else:
				gender = 'F'
				genderSet = 1
		elif itemCat[it] == 'G':
			if genderSet == 1 and gender != 'G':
				gender = 'N'
				break
			else:
				gender = 'G'
				genderSet = 1
		elif itemCat[it] == 'B':
			if genderSet == 1 and gender != 'B':
				gender = 'N'
				break
			else:
				gender = 'B'
				genderSet = 1
	userGender[u] = gender
	if gender == 'M':
		M += 1
	elif gender == 'F':
		F += 1
	elif gender == 'G':
		G += 1
	elif gender == 'B':
		B += 1
	else:
		N += 1

#BASELINES
mostPopular = [(itemCount[x], x) for x in itemCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
  count += ic
  return1.add(i)
  if count > totalPurchases/1.5: break
###--MAKE PREDICTIONS
yes = 0
no = 0


mF = 0
tM = 0
fM = 0
tF = 0
predictions = open("predictions_Purchase.txt", 'w')
for l in open("pairs_Purchase.txt"):
	flag = 0
	if l.startswith("userID"):
		#header
		predictions.write(l)
		continue
	u,i = l.strip().split('-')
	
	if i in return1:
		flag = 1
	
	#Check Jaccard
	for m in dic2[i]:
		card = jac(u, m, userItem)
		if card > 0.0:
			flag = 1
	
	#Check if they purchase from that category?
	
	#Pearson Algorithm
	if pearson(u, i):
		flag = 1
	else:
		hng = 0.5
		
	#Cosine Algorithm
	
	
	#Check Cat
	if userGender[u] == 'M':
		tM += 1
		if itemCat[i] == 'F':
			flag = 0
			mF += 1
	#if userGender[u] == itemCat[i]:
	#	flag = 1
	if userGender[u] == 'F':
		tF += 1
		if itemCat[i] == 'M':
			flag = 0
			fM += 1
			
	if hng == 1:
		flag = 0

	if flag == 1:
		yes += 1
		predictions.write(u + '-' + i + ",1\n")
	else:
		no += 1
		predictions.write(u + '-' + i + ",0\n")

predictions.close()

print yes
print no
##


