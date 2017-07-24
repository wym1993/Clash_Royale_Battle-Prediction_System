from numpy import loadtxt
import numpy as np
import csv
from xgboost import XGBClassifier
from matplotlib import pyplot
import pandas as pd

def feaImpPlot(featureMatrix, labMatrix):
	model = XGBClassifier()
	model.fit(featureMatrix, labMatrix)
	# feature importance
	print model.feature_importances_
	pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
	pyplot.show()

featureNum = 100;
data = loadtxt('card+cost+type+rarity+pat+pop.csv', delimiter=",")
featureMatrix = data[:,:-1];
labMatrix = data[:,-1];
print featureMatrix.shape, labMatrix.shape

print ''
print 'feature importance plot';
feaImpPlot(featureMatrix, labMatrix)