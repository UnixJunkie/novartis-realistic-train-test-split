#this is python 3.6!
"""
The MIT License

Copyright (c) 2011 Dominic Tarr

Permission is hereby granted, free of charge, 
to any person obtaining a copy of this software and 
associated documentation files (the "Software"), to 
deal in the Software without restriction, including 
without limitation the rights to use, copy, modify, 
merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom 
the Software is furnished to do so, 
subject to the following conditions:

The above copyright notice and this permission notice 
shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR 
ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import argparse
import sys
#from scipy.spatial.distance import *

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import PandasTools


#This program will take a spreadsheet with smiles strings and add three columns to it:
# 1. Cluster number (starts with 1)
# 2. Number of compounds per cluster
# 3. Training or Test set allocation of this compound (default is 75% to train)
# Valery Polyakov and Xiangwei Zhu (2019)
#usage:
#	initiate the environment
#	python clustering_split.py -i inputFileName -o traingingFileName -s 0-based_smiles_column_index [-f 0.75] [-b True]
#example:
#   time python clustering_split.py -i input4clustering_split.txt -o clustering_results.csv -s 3 -b True
#	
# initiate the parser
parser = argparse.ArgumentParser(description='Program for splitting into test and traing sets:', epilog="for pQSAR paper June 2019")

# add long and short arguments
parser.add_argument("--input", "-i", help="File with compounds. Must contain a smiles column. A 0-based index of this column must be included as a parameter under -s option.")
parser.add_argument("--output", "-o", help="File name for results data")
parser.add_argument("--boolean", "-b", help="Is to split clusters to keep exact test to training ratio", default=True)
parser.add_argument("--trainingFraction", "-f", help="Fraction of the whole dataset to train the model", type=float, default=0.75)
parser.add_argument("--smilesColumnNumber", "-s", help="0-based smiles column number in the input file", type=int)

# read arguments from the command line
args = parser.parse_args()

# check for --input
if not args.input:
	print("Error: no input file provided")
	sys.exit(1)

# check for --output
if not args.output:
	print("Error: no file to save training results")
	sys.exit(1)

# check for --
if not args.smilesColumnNumber:
	print("Error: Smiles column number is missing")
	sys.exit(1)

def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

class FP:
  def __init__(self, fp):
        self.fp = fp
  def __str__(self):
      return self.fp.__str__()

def computeFP(x):
	#compute depth-2 morgan fingerprint hashed to 1024 bits
	try:
		fp = Chem.GetMorganFingerprintAsBitVect(x,2,nBits=1024)
		res = np.zeros(len(fp),np.int32)
		#convert the fingerprint to a numpy array and wrap it into the dummy container
		DataStructs.ConvertToNumpyArray(fp,res)
		return FP(res)
	except:
		print("FPs for a structure cannot be calculated")
		return None

#Getting model/dataset stats for challenged set
def dists_yield(fps, nfps):
	# generator
	for i in range(1, nfps):
		yield [1-x for x in DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])]

#def ClusterData(fps, nPts, distThresh, isDistData=False, reordering=False):
def ClusterData(fps, nPts, distThresh, reordering=False):
	"""	clusters the data points passed in and returns the list of clusters

		**Arguments**

			- data: a list of items with the input data
				(see discussion of _isDistData_ argument for the exception)

			- nPts: the number of points to be used

			- distThresh: elements within this range of each other are considered
				to be neighbors
			- reodering: if this toggle is set, the number of neighbors is updated
					 for the unassigned molecules after a new cluster is created such
					 that always the molecule with the largest number of unassigned
					 neighbors is selected as the next cluster center.
		**Returns**
			- a tuple of tuples containing information about the clusters:
				 ( (cluster1_elem1, cluster1_elem2, ...),
					 (cluster2_elem1, cluster2_elem2, ...),
					 ...
				 )
				 The first element for each cluster is its centroid.

	"""
	nbrLists = [None] * nPts
	for i in range(nPts):
		nbrLists[i] = []

	#dmIdx = 0
	dist_fun = dists_yield(fps, nPts)
	for i in range(1, nPts):
		#print(i)
		dists = next(dist_fun)

		for j in range(i):
			#if not isDistData:
			#	dij = EuclideanDist(data[i], data[j])
			#else:
				#dij = data[dmIdx]
			dij = dists[j]
				#dmIdx += 1
			if dij <= distThresh:
				nbrLists[i].append(j)
				nbrLists[j].append(i)

	# sort by the number of neighbors:
	tLists = [(len(y), x) for x, y in enumerate(nbrLists)]
	tLists.sort(reverse=True)

	res = []
	seen = [0] * nPts
	while tLists:
		_, idx = tLists.pop(0)
		if seen[idx]:
			continue
		tRes = [idx]
		for nbr in nbrLists[idx]:
			if not seen[nbr]:
				tRes.append(nbr)
				seen[nbr] = 1
		# update the number of neighbors:
		# remove all members of the new cluster from the list of
		# neighbors and reorder the tLists
		if reordering:
			# get the list of affected molecules, i.e. all molecules
			# which have at least one of the members of the new cluster
			# as a neighbor
			nbrNbr = [nbrLists[t] for t in tRes]
			nbrNbr = frozenset().union(*nbrNbr)
			# loop over all remaining molecules in tLists but only
			# consider unassigned and affected compounds
			for x, y in enumerate(tLists):
				y1 = y[1]
				if seen[y1] or (y1 not in nbrNbr):
					continue
				# update the number of neighbors
				nbrLists[y1] = set(nbrLists[y1]).difference(tRes)
				tLists[x] = (len(nbrLists[y1]), y1)
			# now reorder the list
			tLists.sort(reverse=True)
		res.append(tuple(tRes))
	return tuple(res)

def ClusterFps(fps, method = "Auto"):
	#Cluster size is probably smaller if the cut-off is larger. Changing its values between 0.4 and 0.45 makes a lot of difference
	dists = []
	nfps = len(fps)
	
	if method == "Auto":
		if nfps >= 10000:
			method = "TB"
		else:
			method = "Hierarchy"
	
	if method == "TB":
		#from rdkit.ML.Cluster import Butina
		cutoff = 0.56
		print("Butina clustering is selected. Dataset size is:", nfps)

		cs = ClusterData(fps, nfps, cutoff)
		
	elif method == "Hierarchy":

		import scipy.spatial.distance as ssd
		from scipy.cluster import hierarchy
		print("Hierarchical clustering is selected. Dataset size is:", nfps)

		avClsize = 8

		#Generate dist matrix
		for i in range(0,nfps):
			sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps)
			dists.append([1-x for x in sims])

			#Change format of dist matrix to package-recognizable one
		disArray = ssd.squareform(dists)
		#Build model
		Z = hierarchy.linkage(disArray)
		
		#Cut-Tree to get clusters
		#x = hierarchy.cut_tree(Z,height = cutoff)
		average_cluster_size = avClsize 
		cluster_amount = int( nfps / average_cluster_size )	 # calculate total amount of clusters by (number of compounds / average cluster size )
		x = hierarchy.cut_tree(Z, n_clusters = cluster_amount )		#use cluster amount as the parameter of this clustering algorithm. 
		
		#change the output format to mimic the output of Butina
		x = list(x.transpose()[0])
		cs = []
		for i in range(max(x)+1):
			cs.append([])

		for i in range(len(x)):
			cs[x[i]].append(i)
	return cs

#will use a TAB '\t' as a default separator, but for files ending with .csv, will return comma
def separator4fileName(fileName):
	if '.csv' in fileName:
		separator = ","
	else:
		separator = '\t'
	return separator

def splitWrapper(input, trainingOut, fraction2train, SmilesColumnNo, is2split, clusterMethod = "Auto"):

	try:
		import math

		#read input file in the dataframe
		data = pd.read_csv(input, sep = separator4fileName(input))

		#smiles column name
		smiles = list(data)[SmilesColumnNo]
		
		#just symbolic names
		molecule = 'molecule'
		FP = 'FP'
		Group = 'Group'
		Testing = 'Testing'

		#generate molecules, it it fails, ignore the exception
		try:
			PandasTools.AddMoleculeColumnToFrame(data, smiles, molecule)
		except:
			print("Erroneous exception was raised and captured...")
		
		#remove records with empty molecules
		data = data.loc[data[molecule].notnull()]

		#filter potentially failed fingerprint computations
		data[FP] = [computeFP(m) for m in data[molecule]]
		data = data.loc[data[FP].notnull()]

		#generate array of fingerprints
		fps = [Chem.GetMorganFingerprintAsBitVect(m, 2, nBits = 2048) for m in data[molecule]]
		
		#cluster & sort the clusters by size
		clusters = ClusterFps(fps)
		L = list(clusters)
		L.sort(key=lambda t: len(t), reverse=True)

		data.drop([molecule, FP], axis = 1, inplace = True)
		
		#this is the last compound that should remain in the training set, if we ignore cluster boundaries
		lastTrainingIndex = int(math.ceil(len(data) * fraction2train))
		#print (lastTrainingIndex)
		
		#I will create a new dataframe
		clustered = None

		#Sequential cluster number
		clusterNo = 0

		#keeps track of the number of processed molecules
		molCount = 0

		#this loop will allocate the biggest clusters to the training set and the smalles to the test set
		#without breaking the cluster boundaries
		for cluster in L:
			
			#consecutive cluster numbers starting with 1
			clusterNo = clusterNo + 1			
			#print("Cluster: %i, Molecules: %i" % (clusterNo, len(cluster)))

			#get all molecules by indexes belongin to this cluster
			try:
				oneCluster = data.iloc[list(cluster)].copy()
			except:
				print("Wrong indexes in Cluster: %i, Molecules: %i" % (clusterNo, len(cluster)))
				continue
			
			#two columns will be added: cluster number and the number of molecules in this cluster
			oneCluster.loc[:,'ClusterNo'] = clusterNo
			oneCluster.loc[:,'MolCount'] = len(cluster)

			#the whole cluster belongs to training if at least one molecule is below the last index -
			# one more column will be added designating trarining or test set
			if (molCount < lastTrainingIndex) or (clusterNo < 2):
				oneCluster.loc[:,Group] = 'Training'
			else:
				oneCluster.loc[:,Group] = Testing
			
			#this will be taken into account with the next iteration in order to 
			# allocate all molecules of the cluster to training, if at least one molecule in in training set
			molCount += len(cluster)
			
			#add new cluster to the processed results
			clustered = pd.concat([clustered, oneCluster], ignore_index = True)

		#if we need to keep test/training ration exact, we may assign some compounds from the border line cluster to two different categories
		if is2split == True:
			print("Adjusting test to train ratio. It may split one cluster")
			#everything after the last molecule will go to the test set
			clustered.loc[lastTrainingIndex + 1:, Group] = Testing

		#write to file
		clustered.to_csv(trainingOut, mode = "w", index = False, sep = separator4fileName(trainingOut))
		print("Clustering finished. Training set size is %i, Test set size is %i, Fraction %.2f" % 
			(len(clustered.loc[clustered[Group] != Testing]), len(clustered.loc[clustered[Group] == Testing]), len(clustered.loc[clustered[Group] == Testing])/len(clustered)))
	
	#sometimes clustering is a long process, so we can iterrupt
	except KeyboardInterrupt:
			print("Clustering interrupted.")

	return 0

#stand alone clustering call

input = args.input
output = args.output

smilesCoNo = args.smilesColumnNumber
is2split = str2bool(args.boolean)

print(" Input: %s\n Training file name: %s\nFraction to train: %.2f\n Split Clusters: %r\n 0-based smiles column number: %i" % 
	(input, output, args.trainingFraction, is2split, smilesCoNo))

sys.exit(splitWrapper(input, output, args.trainingFraction, smilesCoNo, is2split))
sys.exit(1)
