from numpy import loadtxt
import numpy as np
import csv
from xgboost import XGBClassifier
from matplotlib import pyplot
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from scipy.spatial.distance import cosine
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import MultinomialNB
import random
import NN

def readFeature(filename):
	'''
	Read feature from file into a matrix
	'''
	with open(filename,'rU') as csvfile:
		reader=csv.reader(csvfile)
		S = [[str(i) for i in r] for r in reader]
		S = np.matrix(S);
		featureMatrix = S[:,:-1];
		labMatrix = S[:,-1];
	return featureMatrix, labMatrix;

def feaImpPlot(featureMatrix, labMatrix):
	'''
	Plot the feature matrix
	'''
	model = XGBClassifier()
	model.fit(featureMatrix, labMatrix)
	# feature importance
	print model.feature_importances_
	pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
	pyplot.show()

def xgboostEva(featureMatrix, labMatrix, feaLen):
	'''
	Use XGBoost to evalute the feature matrix
	'''
	model = XGBClassifier()
	model.fit(featureMatrix, labMatrix)
	importance = np.array(model.feature_importances_)
	mostimport = importance.argsort()[-feaLen:][::-1];
	return mostimport;

def univarEva(featureMatrix, labMatrix, feaLen):
	test = SelectKBest(score_func=chi2, k=feaLen)
	fit = test.fit(pd.DataFrame(data = featureMatrix), pd.DataFrame(data = labMatrix));
	a = np.array(fit.scores_)
	featuresIdx = [];
	for item in a.argsort()[:][::-1]:
		if a[item]>=0:
			featuresIdx.append(item);
			#print item, a[item]
		if len(featuresIdx)==feaLen:
			break;
	return featuresIdx;

def algoImp(newFeatureMatrix, labMatrix, algorList):
	'''
	Main program for multiple algorithms
	'''
	(row, col) = newFeatureMatrix.shape
	split = int(row*0.8);
	for algor in algorList:
		if algor=='RF':
			clf = RandomForestClassifier(n_estimators=25,warm_start=True,oob_score=True)
			clf.fit(newFeatureMatrix[:split, :], labMatrix[:split])
			pred = clf.predict(newFeatureMatrix[split:,:])
			score = metrics.accuracy_score(labMatrix[split:], pred)
			print ("Random Forest test accuracy is:   %0.3f" % score)

		if algor == 'LSVM':
			clf = SVC(kernel = 'linear').fit(newFeatureMatrix[:split, :], labMatrix[:split])
			accuracy = clf.score(newFeatureMatrix[split:,:],labMatrix[split:])
			print ('Linear SVM test accuracy is',accuracy)	

		if algor == 'PSVM':
			clf = SVC(kernel = 'poly').fit(newFeatureMatrix[:split, :], labMatrix[:split])
			accuracy = clf.score(newFeatureMatrix[split:,:],labMatrix[split:])
			print ('Linear SVM test accuracy is',accuracy)	

		if algor == 'Ada':
			ada = AdaBoostClassifier()
			ada.fit(newFeatureMatrix[:split, :], labMatrix[:split]);
			print 'Adaboost test accuracy is ', ada.score(newFeatureMatrix[split:,:],labMatrix[split:])

		if algor == 'LR':
			regre = LogisticRegression();
			regre.fit(newFeatureMatrix[:split,:], labMatrix[:split]);
			print 'Logistic Regression test accuracy is ', regre.score(newFeatureMatrix[split:,:], labMatrix[split:]);

		if algor == 'NB':
			nb = MultinomialNB();
			nb.fit(newFeatureMatrix[:split,:], labMatrix[:split]);
			print 'Naive Bayes test accuracy is ', nb.score(newFeatureMatrix[split:,:], labMatrix[split:]);

		#if algor == 'DT':

		if algor == 'NN':
			nn = MLPClassifier()
			nn.fit(featureMatrix[:split, :], labMatrix[:split])
			print 'Neural Network test accuracy is ', nn.score(featureMatrix[split:,:],labMatrix[split:])

		if algor == 'ADDT':
			regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=100)
			regr.fit(newFeatureMatrix[:split, :], labMatrix[:split])
			
			pred = regr.predict(newFeatureMatrix[split:, :])
			for i in range(len(pred)):
				if pred[i]>0:
					pred[i]=1;
				else:
					pred[i]=0;
			
			score = metrics.accuracy_score(labMatrix[split:], pred)
			print ("Boosting w/ DT Test accuracy is:   %0.3f" % score)



featureNum = 150;
data = loadtxt('card+cost+type+rarity+pat+pop.csv', delimiter=",")
print data.shape
featureMatrix = data[:,:1004];
labMatrix = data[:,-1];
print featureMatrix.shape, labMatrix.shape

#print ''
#print 'feature importance plot';
#feaImpPlot(featureMatrix, labMatrix)

print ''
print 'XGBoost Analysis'
mostimport = xgboostEva(featureMatrix, labMatrix, featureNum);
"""
print 'Univariate feature selection'
featuresIdx = univarEva(featureMatrix, labMatrix, featureNum)

newFeatureIDX = featuresIdx;
for item in mostimport:
	if item in featuresIdx:
		newFeatureIDX.append(item);
newFeatureIDX = sorted(newFeatureIDX)
print len(newFeatureIDX), len(list(set(newFeatureIDX)))
newFeatureIDX = list(set(newFeatureIDX))
"""
newFeatureIDX = list(mostimport)

#algorList = ['Ada'];
algorList = ['RF', 'LSVM', 'PSVM', 'Ada', 'LR', 'NB', 'NN', 'ADDT'];
#algorList = ['ADDT'];
newFeatureMatrix = featureMatrix[:,newFeatureIDX]
algoImp(newFeatureMatrix, labMatrix, algorList)

print 'finished'

