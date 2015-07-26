import gzip
import operator
import numpy
from collections import defaultdict
import urllib

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

true = True
false = False

print "Reading data..."
userD = list(parseData("yelp_academic_dataset_user.json"))
businessD = list(parseData("yelp_academic_dataset_business.json"))
reviewD = list(parseData("yelp_academic_dataset_review.json"))
checkinD = list(parseData("yelp_academic_dataset_checkin.json"))
print "done"

#############################################################################

# Get all the business IDs
businessIDs = []

for i in businessD:
	businessIDs.append([i['business_id'], i['city'], i['neighborhoods']])

# for i in businessD:
# 	businessLocation.append([i['city'], i['neighborhoods']])
"""
# Get number of checkins for each business
checkinDict = {}

for i in checkinD:
	checkinDict[i['business_id']] = i['checkin_info']
"""

# list of businesses with checkins
checkinBiz = []
# list of dictionaries of checkins per business
checkinDict = []
# sum of all checkins per business
checkinSum = []
# Used for the first 5000 of most popular businesses
bizTownSorted = []
# To get the most popular neighborhoods
neighborhoods = []

# Add checkins to the list
for i in checkinD:
	checkinDict.append(i['checkin_info'])

# Add businesses to the list
for i in checkinD:
	checkinBiz.append(i['business_id'])

# Go through each dictionary in the list
for x in range(0, len(checkinDict)):
	# Sum values in each dictionary
	checkinSum.append(sum(checkinDict[x].values()))

# Merge businesses and number of checkins to dictionary
dicBizSums = dict(zip(checkinBiz, checkinSum))

# Used to sort by the value of the key
sortedCheckins = sorted(dicBizSums.items(), key=operator.itemgetter(1))

# Reverse to get the most popular at the beginning of the dictionary
sortedCheckins.reverse()

for i in sortedCheckins[:5000]:
	bizTownSorted.append(i[0])

# Trying to get the city/neighborhoods of every popular business
for i in bizTownSorted:
	if i == businessIDs[i][0]:
		neighborhoods.append(businessIDs[i])
