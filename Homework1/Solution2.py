### Problem 2 Solution
import numpy
import urllib
import scipy.optimize
import random
### from sklearn import svm

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("http://jmcauley.ucsd.edu/cse255/data/amazon/book_descriptions_50000.json"))
print "done"

### Probabililty that a book has category "Romance" (a)

hasRomance = ["Romance" in b['categories'] for b in data]
hasRomance = sum(hasRomance) * 1.0 / len(hasRomance)

# Probabililty that a book doesn't have category "Romance"

negHasRomance = 1 - hasRomance

# Probability that book mentions "love" in description given it is Romance (b)

loveInDescription = ['love' in b['description'] for b in data if "Romance" in b['categories']]
loveInDescription = sum(loveInDescription) * 1.0 / len(loveInDescription)

### Problem 2.2

############### CORRECT SOLUTION ###############

# p(Romance book)
prior = ["Romance" in b['categories'] for b in data]
prior = sum(prior) * 1.0 / len(prior)

# p(isn't Romance book)
prior_neg = 1 - prior

# p(mentions 'love' | is romance)
p1 = ['love' in b['description'] for b in data if "Romance" in b['categories']]
p1 = sum(p1) * 1.0 / len(p1)

# p(mentions 'love' | isn't romance)
p1_neg = ['love' in b['description'] for b in data if not("Romance" in b['categories'])]
p1_neg = sum(p1_neg) * 1.0 / len(p1_neg)

# p(mentions 'beaut' | is romance)
p2 = ['beaut' in b['description'] for b in data if "Romance" in b['categories']]
p2 = sum(p2) * 1.0 / len(p2)

# p(mentions 'beaut' | isn't romance)
p2_neg = ['beaut' in b['description'] for b in data if not("Romance" in b['categories'])]
p2_neg = sum(p2_neg) * 1.0 / len(p2_neg)

score1 = prior * p1 * p2
score_neg1 = prior_neg * p1_neg * p2_neg
prediction1 = score1 / score_neg1


print "Value we get from running bayes on the two probabilities: " + str(prediction)

print "The string 'beaut' is more effective than separating 'beauty'/'beautiful' because we need a "
print "mutually exclusive calculation - just use 'beaut'. This is so we can exclude multiple "

### Problem 2.3
# Calculating love not in description

# p(does not mention 'love' | is romance)
pNoLove = ['love' not in b['description'] for b in data if "Romance" in b['categories']]
pNoLove = sum(pNoLove) * 1.0 / len(pNoLove)

# p(does not mention 'love' | is not romance)
pNoLoveNoRomance = ['love' not in b['description'] for b in data if "Romance" not in b['categories']]
pNoLoveNoRomance = sum(pNoLoveNoRomance) * 1.0 / len(pNoLoveNoRomance)

score2 = prior * pNoLove * p2
score_neg2 = prior_neg * pNoLoveNoRomance * p2_neg
prediction2 = score2 / score_neg2

# Calculating beaut not in description

# p(does not mention 'love' | is romance)
pNoBeaut = ['beaut' not in b['description'] for b in data if "Romance" in b['categories']]
pNoBeaut = sum(pNoBeaut) * 1.0 / len(pNoBeaut)

# p(does not mention 'love' | is not romance)
pNoBeautNoRomance = ['beaut' not in b['description'] for b in data if "Romance" not in b['categories']]
pNoBeautNoRomance = sum(pNoBeautNoRomance) * 1.0 / len(pNoBeautNoRomance)

score3 = prior * p1 * pNoBeaut
score_neg3 = prior_neg * p1_neg * pNoBeautNoRomance
prediction3 = score3 / score_neg3

# Calculating beaut and love not in description

score4 = prior * pNoLove * pNoBeaut
score_neg4 = prior_neg * pNoLoveNoRomance * pNoBeautNoRomance
prediction4 = score4 / score_neg4

# Calculate the TP TN FP FN

def number3(datam):
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	for b in datam:
		if 'love' in b['description'] and 'beaut' in b['description']:
			if "Romance" in b['categories']:
				TN += 1
			else:
				FN += 1
		elif 'love' in b['description'] and 'beaut' not in b['description']:
			if "Romance" in b['categories']:
				TN += 1
			else:
				FN += 1
		elif 'love' not in b['description'] and 'beaut' in b['description']:
			if "Romance" in b['categories']:
				TN += 1
			else:
				FN += 1
		elif 'love' not in b['description'] and 'beaut' not in b['description']:
			if "Romance" in b['categories']:
				TN += 1
			else:
				FN += 1
	feat = [TP]
	feat.extend([TN,FP,FN])
	return feat

classifier = number3(data)



