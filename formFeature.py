import numpy as np 
import csv
import readCardInfo
import pandas as pd

def readS(filename):
	'''
	Read file into matrix
	'''
	with open(filename,'rU') as csvfile:
		reader=csv.reader(csvfile)
		S = [[str(i) for i in r] for r in reader]
		S = np.matrix(S)
	return S

def readCARD(filename):
	'''
	Read card info
	'''
	wb = open_workbook(filename)
	items = {}
	for sheet in wb.sheets():
		number_of_rows = sheet.nrows
		number_of_columns = sheet.ncols
		rows = []
		for row in range(1, number_of_rows):
			values = []
			for col in range(number_of_columns):
				value  = str(sheet.cell(row,col).value)
				try:
					value = str(xlrd.xldate_as_tuple(value, 0))
				except ValueError:
					pass
				finally:
					values.append(value)
			item = cardObj(*values)
			items[item.NAME] = item
	return items

def genCardMatrix(DataM, cardInfo):
	'''
	Generate card composition and label matrix
	'''
	cardNum = len(cardInfo);
	row, col = DataM.shape;
	cardMatrix = []
	matchResult = []
	for i in range(row):
		score1 = int(DataM[i,-2])
		score2 = int(DataM[i,-1])
		if score1==score2:
			continue;
		elif score1>score2:
			matchResult.append(np.array(1))
		else:
			matchResult.append(np.array(0));

		line = np.array([0 for p in range(cardNum*2)]);
		for j in range(col-2):
			cardObj = cardInfo[DataM[i,j]]
			if j <8:
				line[int(float(cardObj.ID))-1] = 1#cardObj.NAME.ljust(8)
			else:
				line[int(float(cardObj.ID))-1 + cardNum] = 1#cardObj.NAME.ljust(8)

		cardMatrix.append(line);

	return np.matrix(cardMatrix), np.matrix(matchResult).T;

def genCostMatrix(DataM, cardInfo):
	'''
	Generate cost matrix
	'''
	row, col = DataM.shape;
	level = 6
	costMatrix = []
	for i in range(row):
		score1 = int(DataM[i,-2])
		score2 = int(DataM[i,-1])
		if score1==score2:
			continue;
		ave1, ave2 = getAveCost(np.asarray(DataM[i,0:(col-2)]).ravel(), cardInfo);
		line = np.array([0 for p in range(level*2)]);
		idx1 = int(np.floor(ave1))-1;
		idx2 = int(np.floor(ave2))-1;
		line[idx1] = 1;
		line[idx2+level] = 1;
		costMatrix.append(line);

	return np.matrix(costMatrix);

def getAveCost(line, cardInfo):
	'''
	Calculate average cost for two decks
	'''
	ave1 = 0
	ave2 = 0
	for i in range(8):
		ave1+=float(cardInfo[line[i]].COST)
		ave2+=float(cardInfo[line[i+8]].COST)

	return ave1/8.0, ave2/8.0;

def genTypeMatrix(DataM, cardInfo):
	'''
	Generate card type matrix
	'''
	row, col = DataM.shape;
	typeMatrix = [];
	typeList = ['Troops', 'Spells', 'Buildings'];
	for i in range(row):
		score1 = int(DataM[i,-2])
		score2 = int(DataM[i,-1])
		if score1==score2:
			continue;
		line = np.array([0 for p in range(9*len(typeList)*2)]);
		type1, type2 = getType(np.asarray(DataM[i,0:(col-2)]).ravel(), cardInfo, typeList);
		for i in range(len(type1)):
			idx1 = type1[i]+i*9;
			idx2 = type2[i]+(i+len(typeList))*9;
			line[idx1] = 1;
			line[idx2] = 1;
		typeMatrix.append(line);

	return np.matrix(typeMatrix);

# calculate type information for two decks
def getType(line, cardInfo, typeList):
	type1 = [0 for i in range(len(typeList))];
	type2 = [0 for i in range(len(typeList))];
	for i in range(8):
		tmpT1 = cardInfo[line[i]].TYPE;
		tmpT2 = cardInfo[line[i+8]].TYPE;
		type1[typeList.index(tmpT1)]+=1;
		type2[typeList.index(tmpT2)]+=1;

	return type1, type2;

# Generate cards rarity matrix
def genRarityMatrix(DataM, cardInfo):
	row, col = DataM.shape;
	RarityMatrix = [];
	rarityList = ['common', 'Rare', 'Epic', 'Legendary'];
	for i in range(row):
		score1 = int(DataM[i,-2])
		score2 = int(DataM[i,-1])
		if score1==score2:
			continue;
		line = np.array([0 for p in range(9*len(rarityList)*2)]);
		rarity1, rarity2 = getRarity(np.asarray(DataM[i,0:(col-2)]).ravel(), cardInfo, rarityList)
		for i in range(len(rarity1)):
			idx1 = rarity1[i]+i*9;
			idx2 = rarity2[i]+(i+len(rarityList))*9;
			line[idx1] = 1;
			line[idx2] = 1;
		RarityMatrix.append(line);

	return np.matrix(RarityMatrix);

# Calculate cards rarity for two decks
def getRarity(line, cardInfo, rarityList):
	rarity1 = [0 for i in range(len(rarityList))];
	rarity2 = [0 for i in range(len(rarityList))];
	for i in range(8):
		tmpR1 = cardInfo[line[i]].RARITY;
		tmpR2 = cardInfo[line[i+8]].RARITY;
		rarity1[rarityList.index(tmpR1)]+=1;
		rarity2[rarityList.index(tmpR2)]+=1;

	return rarity1, rarity2

# Generate frequent pattern matrix
def genPatternMatrix(DataM, pattern):
	row, col = DataM.shape;
	numPat = len(pattern);
	patMatrix = [];
	for i in range(row):
		score1 = int(DataM[i,-2])
		score2 = int(DataM[i,-1])
		if score1==score2:
			continue;
		tmpPat = [1 for m in range(numPat*2)];
		for j in range(numPat):
			pat = pattern[j];
			player1 = np.asarray(DataM[i, 0:8]).ravel();
			player2 = np.asarray(DataM[i, 8:16]).ravel();
			if not checkSub(player1, pat):
				tmpPat[j*2] = 0;
			if not checkSub(player2, pat):
				tmpPat[j*2+1] = 0;
			
		patMatrix.append(tmpPat);

	return np.matrix(patMatrix);

# Check subarray of an array
def checkSub(arr, subArr):
	if set(arr)-set(subArr)!=set([]) and set(subArr)-set(arr)==set([]):
		return True;
	else:
		return False;

# Generate popular card deck matrix
def genPopMatrix(DataM, popDeck):
	rowDM, colDM = DataM.shape;
	rowPop, colPop = popDeck.shape;
	popMatrix = [];
	for i in range(rowDM):
		score1 = int(DataM[i,-2])
		score2 = int(DataM[i,-1])
		if score1==score2:
			continue;

		print i
		tmp = [0 for i in range(rowPop*2*9)]
		
		player1 = np.asarray(DataM[i, 0:8]).ravel();
		player2 = np.asarray(DataM[i, 8:16]).ravel();
		for j in range(rowPop):
			num1 = checkDis(player1, popDeck[j,:]);
			num2 = checkDis(player2, popDeck[j,:]);
			tmp[j*8+num1] = 1;
			tmp[(j+rowPop)*8+num2] = 1;
		popMatrix.append(tmp);

	return np.matrix(popMatrix);

# Calculate deck distance 
def checkDis(deck1, deck2):
	num = 0;
	for card in deck1:
		if card in deck2:
			num+=1;
	return num;

# read frequent pattern from the file
def readPattern(filename, threshold):
	patMatrix = []
	with open(filename) as f:
		allPattern = f.readlines()[2:];
		for pattLine in allPattern:
			line = pattLine.split(' ');
			if int(line[0]) < threshold:
				continue;
			patt = line[2:];
			patt[0] = patt[0][1:];
			patt[-1] = patt[-1][:-2]
			patMatrix.append(patt);

	return patMatrix;

# Write the feature matrix into a file for future use
def writeFeatureMatrix(featureMatrix, filename):
	file = open(filename, 'w');
	wr = csv.writer(file, dialect='excel');
	(row, col) = featureMatrix.shape;
	for i in range(row):
		print i
		tmpArr = np.asarray(featureMatrix[i,:]).ravel();
		wr.writerow(tmpArr)


def main():
	# Read necessary data matrix into the program
	xlsxName = 'cards.xlsx'
	dataName = 'data_all.csv';
	popDeckName = 'popDeck.csv';
	patternName = 'pattern.txt';
	patternThresh = 200;
	DataM = readS(dataName)
	popDeck = readS(popDeckName);
	pattern = readPattern(patternName, patternThresh);
	cardInfo = readCardInfo.readCARD(xlsxName)
	cardMatrix, matchResult = genCardMatrix(DataM, cardInfo);
	print 'cardMatrix', cardMatrix.shape
	print 'matchResule', matchResult.shape;

	# Generate feature matrix
	feaComb = ['card', 'cost', 'type', 'rarity', 'pat', 'pop'];
	featureMatrix = np.empty([matchResult.shape[0],0]);
	print 'Initial',featureMatrix.shape
	for fea in feaComb:
		# Geneate card composition feature
		if fea == 'card':
			print 'cardMatrix', cardMatrix.shape
			print 'matchResule', matchResult.shape;
			featureMatrix = np.append(featureMatrix, cardMatrix, axis=1);

		# Generate average elixir cost feature
		if fea == 'cost':
			costMatrix = genCostMatrix(DataM, cardInfo);
			print 'costMatrix', costMatrix.shape;
			featureMatrix = np.append(featureMatrix, costMatrix, axis=1);

		# Generate card type feature
		if fea == 'type':
			typeMatrix = genTypeMatrix(DataM, cardInfo);
			print 'typeMatrix', typeMatrix.shape
			featureMatrix = np.append(featureMatrix, typeMatrix, axis=1);

		# Generate card rarity feature
		if fea == 'rarity':
			RarityMatrix = genRarityMatrix(DataM, cardInfo);
			print 'RarityMatrix', RarityMatrix.shape
			featureMatrix = np.append(featureMatrix, RarityMatrix, axis=1);

		# Generate frequent pattern feature
		if fea == 'pat':
			patMatrix = genPatternMatrix(DataM, pattern);
			print 'PatMatrix', patMatrix.shape
			featureMatrix = np.append(featureMatrix, patMatrix, axis=1);

		# Generate popular card deck comparison feature
		if fea == 'pop':
			popMatrix = genPopMatrix(DataM, popDeck);
			print 'PopMatrix', popMatrix.shape
			featureMatrix = np.append(featureMatrix, popMatrix, axis=1);

	# Append the label
	featureMatrix = np.append(featureMatrix, matchResult, axis=1);
	print 'featureMatrix', featureMatrix.shape
	# Write the feature matrix to file for future use
	writeName = '+'.join(feaComb)+'.csv';
	writeFeatureMatrix(featureMatrix, writeName);

	print 'finished';

if __name__ == "__main__":
    #main();
    pass


"""
costMatrix = genCostMatrix(DataM, cardInfo);
print 'costMatrix', costMatrix.shape;
typeMatrix = genTypeMatrix(DataM, cardInfo);
print 'typeMatrix', typeMatrix.shape
RarityMatrix = genRarityMatrix(DataM, cardInfo);
print 'RarityMatrix', RarityMatrix.shape
patMatrix = genPatternMatrix(DataM, pattern);
print 'PatMatrix', patMatrix.shape

popMatrix = genPopMatrix(DataM, popDeck);
print 'PopMatrix', popMatrix.shape
featureMatrix = popMatrix
#featureMatrix = cardMatrix;
#featureMatrix = patMatrix
#featureMatrix = np.append(featureMatrix, costMatrix, axis=1);
#featureMatrix = np.append(featureMatrix, typeMatrix, axis=1);
#featureMatrix = np.append(featureMatrix, RarityMatrix, axis=1);
featureMatrix = np.append(featureMatrix, patMatrix, axis=1);
featureMatrix = np.append(featureMatrix, matchResult, axis=1);
print 'featureMatrix', featureMatrix.shape
writeFeatureMatrix(featureMatrix)
"""
# Implementation of nn algorithm
"""
batch_size = 10;
learning_rate = 0.1;
activation_function = 'tanh';
hidden_layer_width = 10;
domain = 'mnist';
net = NN.create_NN(col, domain, batch_size, learning_rate, activation_function, hidden_layer_width)
net.train(newTrainMatrix);
print 'accuracy ' + str(net.evaluate(newTestMatrix));

"""




