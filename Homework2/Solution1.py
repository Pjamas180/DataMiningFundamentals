# Homework 2 Part 1

import numpy
import urllib
import scipy.optimize
import random
from sklearn.decomposition import PCA
from collections import defaultdict

### PCA on beer reviews ###

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))
print "done"

X = [[x['review/overall'], x['review/taste'], x['review/aroma'], x['review/appearance'], x['review/palate']] for x in data]

# Find the 5-d mean of X

a = sum([x[0] for x in X])
b = sum([x[1] for x in X])
c = sum([x[2] for x in X])
d = sum([x[3] for x in X])
e = sum([x[4] for x in X])

fiveDMean = [a/50000, b/50000, c/50000, d/50000, e/50000]

a = numpy.array(X)
fiveDMean = a.mean(axis=0)

# Find the Reconstruction Error
Y = sum(sum((fiveDMean - X)**2))

# Centroid Algorithm

# Initialize Centroids
centroids = [[0,0,0,0,1], [0,0,0,1,0]]

def distance(x, y):
	summed = 0
	for x1, y1 in zip(x, y):
		a = (x1-y1)**2
		summed = summed + a
	return summed

def centroidAlg(centroid, datum):
	Y = []
	count0 = 0
	count1 = 0
	centroid0Array = [centroid[0]]
	centroid1Array = [centroid[1]]
	newDataSet = []
	for d in datum:
		x1 = distance(d,centroid[0])
		x2 = distance(d,centroid[1])
		if (x1 < x2):
			centroid0Array.append(d)
			Y.append(0)
			count0 += 1
			x = [x - y for x,y in zip(d, centroid[0])]
			newDataSet.append(centroid[0])
		else:
			centroid1Array.append(d)
			Y.append(1)
			count1 += 1
			x = [x - y for x,y in zip(d, centroid[0])]
			newDataSet.append(centroid[1])
	a = numpy.array(centroid0Array)
	b = numpy.array(centroid1Array)
	centroid[0] = a.mean(axis=0)
	centroid[1] = b.mean(axis=0)
	return [centroid,Y,count0,count1,newDataSet]

centroidReturn0 = centroidAlg(centroids, X)
centroidReturn1 = centroidAlg(centroidReturn0[0], X)

def comparison(x, y):
	for x1, y1 in zip(x,y):
		if (x1 != y1):
			return False
	return True

def repeat(centroids, datum):
	centroidReturn0 = centroidAlg(centroids,X)
	centroidReturn1 = centroidAlg(centroidReturn0[0],X)
	while True:
		if(comparison(centroidReturn0[1],centroidReturn1[1])):
			return centroidReturn0[0]
		else:
			centroidReturn0 = centroidAlg(centroidReturn0[0],X)
			if(comparison(centroidReturn1[1],centroidReturn0[1])):
				return centroidReturn1[0]
			else:
				centroidReturn1=centroidAlg(centroidReturn1[0],X)

result = repeat(centroids,X)

centroids = [[4.17993, 4.23675, 4.14107, 4.08866, 4.12518], [3.09862, 3.06899, 3.14020, 3.38222, 3.11332]]

# Number 5 - 36534, 13466 obtained from count0 and count1 from centroidAlg

t1_6 = sum([sum([(c[i] - x[i]) ** 2 for i in range(5)]) for c, x in zip(result[4], X)])

# Number 6 - Reconstruction Error = 63420.42653275129





