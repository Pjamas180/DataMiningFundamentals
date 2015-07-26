# Homework 2 Part 2

import numpy
import urllib
import scipy.optimize
import random
from sklearn.decomposition import PCA
from collections import defaultdict

### Network visualization ###
import networkx as nx
import matplotlib.pyplot as plt


edges = set()
nodes = set()
for edge in urllib.urlopen("http://jmcauley.ucsd.edu/cse255/data/facebook/egonet.txt", 'r'):
  x,y = edge.split()
  x,y = int(x),int(y)
  edges.add((x,y))
  edges.add((y,x))
  nodes.add(x)
  nodes.add(y)

len(edges)
len(nodes)

G = nx.Graph()
for e in edges:
  G.add_edge(e[0],e[1])
nx.draw(G)
plt.show()
plt.clf()

connectedComponents = list(nx.connected_components(G))
len(connectedComponents)
len(connectedComponents[0])

### Find all 3 and 4-cliques in the graph ###
cliques3 = set()
cliques4 = set()
for n1 in nodes:
  for n2 in nodes:
    if not ((n1,n2) in edges): continue
    for n3 in nodes:
      if not ((n1,n3) in edges): continue
      if not ((n2,n3) in edges): continue
      clique = [n1,n2,n3]
      clique.sort()
      cliques3.add(tuple(clique))
      for n4 in nodes:
        if not ((n1,n4) in edges): continue
        if not ((n2,n4) in edges): continue
        if not ((n3,n4) in edges): continue
        clique = [n1,n2,n3,n4]
        clique.sort()
        cliques4.add(tuple(clique))

c = list(nx.k_clique_communities(G,4))
len(c)
len(c[0])

# Given a clique size: 4
# Initialize every clique as it's own community - cliques4()

for x in cliques4:
	for y in cliques4:
		for z in cliques3:
			if (z in x and z in y):
				x1 = x + y
				y = y + x
				x = x1







