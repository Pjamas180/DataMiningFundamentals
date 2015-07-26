# Homework 3

import numpy
import urllib
import scipy.optimize
import random
import math

from sklearn.decomposition import PCA
from collections import defaultdict

import json
import gzip

# JSON divided into itemID, rating, helpful {nHelpful, outOf} reviewText

def readGz(f):
	for l in gzip.open(f):
		yield eval(l)

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

data = list(parseData("train.json"))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~ TASK 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Using the training data ('train.json.gz'), fit a simple 
# predictor of the form rating(user,item) ~= a.
# Report a and the MSE of your predictor against the test data 
# ('labeled.Rating.txt')

allRatings = []
userRating = defaultdict(list)
distinct_users = []
distinct_items = []

def addToList(self,str_to_add):
	if str_to_add not in self:
		self.append(str_to_add)

for l in readGz("train.json.gz"):
	user,item = l['reviewerID'],l['itemID']
	allRatings.append((l['rating'],user,item))
	userRating[user].append(l['rating'])
	addToList(distinct_users,user)
	addToList(distinct_items,item)


averageRate = sum([x for (x,y,z) in allRatings]) * 1.0 / len(allRatings)
userRate = {}
for u in userRating:
	userRate[u] = sum

# Value of a on the training set.
print "average rate is 3.60335"

# Since the RMSE is the standard deviation of the formula, then we
# just take the square of the RMSE to get MSE
varRating = numpy.var(allRatings)

print "MSE is 2.6089787775019451" # this is the MSE of the training set...

# Find the MSE of the predictor against the test data.
testRatings = []
users = []
items = []
for l in open("labeled_Rating.txt"):
	u,i,rating = l.strip().split(" ")
	testRatings.append((rating, u, i))
	users.append(u)
	items.append(i)

testRate = sum([(float(x) - averageRate)**2 for (x,y,z) in testRatings]) * 1.0 / len(testRatings)

print "MSE on test set is 2.617831892499797"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~ TASK 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Fit a predictor of the form  
# rating(user,item) ~= a + b(user) + b(item)

# 1 + number of users + number of items
# rating(user,item ~= averageRating + b(user) + b(item))

# Calculate for Item: I102776733 and User: U566105319

# Initial Values
alpha = averageRate
beta_users = {}
beta_items = {}
bUser = 'U566105319'
bItem = 'I102776733'

def calcAlpha(allRatings):
	newAlpha = sum([x - (beta_users[y][0] + beta_items[z][0]) for (x, y, z) in allRatings]) * 1.0 / len(allRatings)
	return newAlpha

def calcBetaU(alpha, buser):
	return sum([float(y)-(alpha+beta_items[x][0]) for (x,y) in beta_users[buser][1]]) * 1.0 / (1+len(beta_users[buser][1]))

def calcBetaI(alpha, bitem):	
	return sum([float(y)-(alpha+beta_users[x][0]) for (x,y) in beta_items[bitem][1]]) * 1.0 / (1+len(beta_items[bitem][1]))


for x in distinct_users:
	beta_users[x] = [0,[]]

for x in distinct_items:
	beta_items[x] = [0,[]]

for l in readGz("train.json.gz"):
	user,item,rating = l['reviewerID'],l['itemID'],l['rating']
	addToList(beta_users[user][1],(item,rating))
	addToList(beta_items[item][1],(user,rating))

for x in beta_users:
	beta_users[x][0] = calcBetaU(alpha,x)

for x in beta_items:
	beta_items[x][0] = calcBetaI(alpha,x)

newArray = []
for l in readGz("train.json.gz"):
	user,item,rating = l['reviewerID'],l['itemID'],l['rating']
	newError = alpha + beta_users[user][0] + beta_items[item][0]
	newArray.append((newError - float(rating))**2)

newArraySum = sum(newArray)

# Calculating the Regularizer
betaUArray = []
betaIArray = []
for x in beta_users:
	betaUArray.append(beta_users[x][0]**2)
newBetaUArraySum = sum(betaUArray)
for x in beta_items:
	betaIArray.append(beta_items[x][0]**2)
newBetaIArraySum = sum(betaIArray)
mse = newArraySum + newBetaUArraySum + newBetaIArraySum


# Calculating the error
oldmse = 0.0
while math.fabs(oldmse - mse) > 0.0001*oldmse:
	# Calculating new alpha
	alpha = calcAlpha(allRatings)
	# Calculating new beta_users and beta_items
	for x in beta_users:
		beta_users[x][0] = calcBetaU(alpha,x)
	for x in beta_items:
		beta_items[x][0] = calcBetaI(alpha,x)
	# Optimization - calculating error	
	newArray = []
	for l in readGz("train.json.gz"):
		user,item,rating = l['reviewerID'],l['itemID'],l['rating']
		newError = alpha + beta_users[user][0] + beta_items[item][0]
		newArray.append((newError - float(rating))**2)
	newArraySum = sum(newArray)
	# Calculating the Regularizer
	betaUArray = []
	betaIArray = []
	for x in beta_users:
		betaUArray.append(beta_users[x][0]**2)
	newBetaUArraySum = sum(betaUArray)
	for x in beta_items:
		betaIArray.append(beta_items[x][0]**2)
	newBetaIArraySum = sum(betaIArray)
	oldmse = mse
	# Compare this to oldmse to see for convergence.
	mse = newArraySum + newBetaUArraySum + newBetaIArraySum
	print "New mse = " + str(mse) + "Old mse = " + str(oldmse)

# New mse = 105301.873282Old mse = 105312.03445

# b_user for 'U566105319' = -1.0584987923988236
# b_item for 'I102776733' = 0.11761830224986271

# function = alpha + b_user + b_item
alpha = 3.5728290786174632
b_u = -1.0584987923988236
b_i = 0.11761830224986271

rating = alpha + b_u + b_i
# rating = 2.6319485884685023

testRate = sum([(float(x) - (3.5728290786174632 + beta_users[y][0] + beta_items[z][0]))**2 for (x,y,z) in testRatings if y in beta_users and z in beta_items]) * 1.0 / len(testRatings)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~ TASK 3 ~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Jaccard Similarity: 0.75
# User with highest similarity to U622491081: U359587607, U096951499, U687939146, U296575297, U387971231, U300899166

##### BEST SOLUTION 
# Build a dictionary
from collections import defaultdict
d = defaultdict(list)

for l in readGz("train.json.gz"):
	#for x in u2:
	#if (l['itemID'] == x):
	d[l['reviewerID']].append(l['itemID'])


def jaccard_similarity(user_items, u1, u2):
	return len(set(user_items[u1]).intersection(user_items[u2]))*1.0/len(set(user_items[u1]).union(user_items[u2]))

u1 = "U622491081"
jaccardAll = [ (x,jaccard_similarity(d, u1, x)) for x in d if x != u1 ]

from operator import itemgetter
maxJac = max(jaccardAll, key=itemgetter(1))[0]



#~~~~~~~~~~~~~~~~~~~~~~~~~~~ TASK 4 ~~~~~~~~~~~~~~~~~~~~~~~~~~~#

avg =[]
for l in data:
	helpful,total = l['helpful']['nHelpful'],l['helpful']['outOf']
	avg.append(float(helpful)/float(total))

# ANSWER PART A
a = sum(avg)/len(avg)  # 0.5331858495042331

# ANSWER PART B
mse = 0
absolute = 0

predicting = []
for l in open("labeled_Helpful.txt"):
	u,i,outof,helpful = l.strip().split(" ")
	predicting.append((a*float(outof),float(helpful)))
	absolute += abs((float(helpful) - a*float(outof)))

mse = sum([(float(helpful)-float(prediction))**2 for (prediction,helpful) in predicting]) * 1.0 / len(predicting)

# mse = 74.4869061806201

test_helpful = [line.strip().split(' ') for line in open("labeled_Helpful.txt")]
mse = numpy.mean([(float(y[3]) - (a * float(y[2]))) ** 2 for y in test_helpful])
t4_2 = mse 
print "MSE: " + str(t4_2)
ae = sum([math.fabs(float(y[3]) - (a * float(y[2]))) for y in test_helpful])
t4_3 = ae
print "Absolute error: " + str(t4_3)

absError = absolute/len(predicting)

# ANSWER PART C

# Define feature matrices for ratings and reviewText
featureMatrix = []
for l in readGz("train.json.gz"):
	rating, review = l['rating'],l['reviewText']
	featureMatrix.append((rating,review))

# ANSWER PART D





