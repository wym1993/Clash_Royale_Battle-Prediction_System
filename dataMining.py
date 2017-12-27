import csv
import numpy as np
import readCardInfo

def readS(filename):
	with open(filename,'rU') as csvfile:
		reader=csv.reader(csvfile)
		S = [[str(i) for i in r] for r in reader]
	return S

def genDistinctComb(DataM, cardInfo):
	'''
	This function find the distinct card combination from data file
	'''
	disCardComb = [];
	for line in DataM:
		comb1 = line[0:7];
		if not comb1 in disCardComb:
			disCardComb.append(comb1);
		comb2 = line[8:15];
		if not comb2 in disCardComb:
			disCardComb.append(comb2);

	return disCardComb;

def countInit(transac):
	'''
	This is to find the initial count from input
	'''
	initCount = {};
	for line in transac:
		for item in line:
			if not item in initCount:
				initCount[str(item)] = 1;
			else:
				initCount[str(item)]+=1;

	return initCount;

def checkSubList(line, comb):
	'''
	This is to check if comb is the sublist of line
	'''
	ls1 = [element for element in line if element in comb]
	return sorted(ls1)==sorted(comb)

def getSup(transac, comb):
	'''
	This is to compute the support of comb in transac
	'''
	sup = 0;
	for line in transac:
		if checkSubList(line, comb):
			sup+=1;

	return sup;

def findFreq(currIt, min_sup, transac):
	'''
	This is to find frequent pattern given current candidate set
	'''
	freq = {};
	consi = []
	for key1 in currIt:
		for key2 in currIt:
			if key2==key1:
				continue;
			l1 = key1.split();
			l2 = key2.split();
			comb = sorted(list(set(l1 + l2)));
			if not comb in consi:
				consi.append(comb);
				string = " ".join(comb);
				tmp = getSup(transac, comb);
				#tmp = min(currIt[key1], currIt[key2]);
				if tmp>=min_sup:
					freq[string] = tmp;

	return freq;

def findFreqPattern(initCount, min_sup, transac):
	'''
	This is the process for finding frequent pattern
	'''
	currCount = {}
	freqPattern = {};
	
	for key in initCount:
		if initCount[key]<min_sup:
			continue;
		"""
		if not initCount[key] in freqPattern:
			freqPattern[initCount[key]] = [key];
		else:
			freqPattern[initCount[key]].append(key);
		"""
		currCount[key] = initCount[key];

	while currCount:
		#print currCount
		freq = findFreq(currCount, min_sup, transac);
		for key in freq:
			if not freq[key] in freqPattern:
				freqPattern[freq[key]] = [key];
			else:
				freqPattern[freq[key]].append(key);

		currCount = freq;

	return freqPattern;

def writeFeatureMatrix(featureMatrix, filename):
	'''
	Write the feature matrix to file
	'''
	file = open(filename, 'w');
	wr = csv.writer(file, dialect='excel');
	for i in range(len(featureMatrix)):
		print i
		wr.writerow(featureMatrix[i])


def main():
	xlsxName = 'cards.xlsx'
	filename = 'data_all.csv';
	min_sup = 500;
	DataM = readS(filename)
	cardInfo = readCardInfo.readCARD(xlsxName)
	#disCardComb = genDistinctComb(DataM, cardInfo)
	#print len(disCardComb)
	#writeFeatureMatrix(disCardComb, 'disCardComb.csv');
	disCardComb = readS('disCardComb.csv');
	print len(disCardComb)
	initCount = countInit(disCardComb)
	print len(initCount)
	freqPattern = findFreqPattern(initCount, min_sup, disCardComb);
	for idx in range(len(disCardComb), min_sup-1, -1):
		if idx in freqPattern:
			ls = sorted(freqPattern[idx]);
			for item in ls:
				print str(idx)+' '+str(len(item.split()))+" ["+str(item)+"]";
	print "finished";

if __name__ == "__main__":
    main();