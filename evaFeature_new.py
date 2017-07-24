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
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import MultinomialNB
import random
import pickle
# import NN

def readFeature(filename):
	with open(filename,'rU') as csvfile:
		reader=csv.reader(csvfile)
		S = [[str(i) for i in r] for r in reader]
		S = np.matrix(S);
		featureMatrix = S[:,:-1];
		labMatrix = S[:,-1];
	return featureMatrix, labMatrix;

featureNum = 50;
data = loadtxt('featureMatrix0427.csv', delimiter=",")
featureMatrix = data[:,:-1];
labMatrix = data[:,-1];
train_data = {}
train_data['x'] = featureMatrix
train_data['y'] = labMatrix
#"begin to save in pickle"
#pickle.dump(train_data, open( "trained_446.p", "wb" ) )
# train_data = pickle.load(open("trained_446.p","rb"))
# featureMatrix = train_data['x']
# labMatrix = train_data['y']
print 'pickle saved'
print featureMatrix.shape, labMatrix.shape
"""
model = XGBClassifier()
model.fit(featureMatrix, labMatrix)
importance = np.array(model.feature_importances_)
mostimport = importance.argsort()[-featureNum:][::-1];

print ''
print 'XGBoost Analysis'
"""
#print mostimport
#pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
#pyplot.show()

"""
# Fit model using each importance as a threshold
thresholds = sort(model.feature_importances_)
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(model, threshold=thresh, prefit=True)
	select_X_train = selection.transform(X_train)
	# train model
	selection_model = XGBClassifier()
	selection_model.fit(select_X_train, y_train)
	# eval model
	select_X_test = selection.transform(X_test)
	y_pred = selection_model.predict(select_X_test)
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
"""
"""
print 'Univariate feature selection'
test = SelectKBest(score_func=chi2, k=featureNum)
fit = test.fit(pd.DataFrame(data = featureMatrix), pd.DataFrame(data = labMatrix));
a = np.array(fit.scores_)
featuresIdx = [];
for item in a.argsort()[:][::-1]:
	if a[item]>=0:
		featuresIdx.append(item);
		#print item, a[item]
	if len(featuresIdx)==featureNum:
		break;
#pyplot.bar(range(len(fit.scores_)), fit.scores_)
#pyplot.show()

newFeatureIDX = featuresIdx;
for item in mostimport:
	if item in featuresIdx:
		newFeatureIDX.append(item);
newFeatureIDX = sorted(newFeatureIDX)
"""
newFeatureMatrix = featureMatrix
(row, col) = newFeatureMatrix.shape
split = int(row*0.8);
"""
# Implementation of nn algorithm
newFeatureMatrix = featureMatrix[:,newFeatureIDX]
(row, col) = newFeatureMatrix.shape
(row2, col2) = featureMatrix.shape;
split = int(row*0.8);
trainMatrix = []
trainMatrix2 = []
for i in range(split):
	trainMatrix.append((np.asarray(newFeatureMatrix[i,:]).ravel(), labMatrix[i]));
	trainMatrix2.append((np.asarray(featureMatrix[i,:]).ravel(), labMatrix[i]))
testMatrix = []
testMatrix2 = [] 
for i in range(split, row):
	testMatrix.append((np.asarray(newFeatureMatrix[i,:]).ravel(), labMatrix[i]));
	testMatrix2.append((np.asarray(featureMatrix[i,:]).ravel(), labMatrix[i]))
print 'Feature selection finished'
print newFeatureMatrix.shape
"""
"""
# Random Forest
print ''
print 'Random Forest'
clf = RandomForestClassifier(n_estimators=25,warm_start=True,oob_score=True)
clf.fit(newFeatureMatrix[:split, :], labMatrix[:split])
pred = clf.predict(newFeatureMatrix[split:,:])
score = metrics.accuracy_score(labMatrix[split:], pred)
print ("Test accuracy is:   %0.3f" % score)

# SVM
print ''
print 'SVM'
clf = SVC(kernel = 'poly').fit(newFeatureMatrix[:split, :], labMatrix[:split])
accuracy = clf.score(newFeatureMatrix[split:,:],labMatrix[split:])
print('SVM accuracy is',accuracy)
"""

"""
# Adaboost
print ''
print 'Adaboost'
ada = AdaBoostClassifier()
ada.fit(newFeatureMatrix[:split, :], labMatrix[:split]);
print ada.score(newFeatureMatrix[split:,:],labMatrix[split:])
"""

#Boosting with Decision tree
#Fit regression model
regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=100)

regr.fit(newFeatureMatrix[:split, :], labMatrix[:split])
print np.unique(labMatrix);

# Predict
pred = regr.predict(newFeatureMatrix[split:, :])
print pred
for i in range(len(pred)):
	if pred[i]>0:
		pred[i]=1;
	else:
		pred[i]=0;
print pred
#print regr.score(newFeatureMatrix[split:,:],labMatrix[split:])
score = metrics.accuracy_score(labMatrix[split:], pred)
print ("Boosting w/ DT Test accuracy is:   %0.3f" % score)

"""
# Linear Regression
print ''
print 'Logistic Regreession';
regre = LogisticRegression();
regre.fit(newFeatureMatrix[:split,:], labMatrix[:split]);
print regre.score(newFeatureMatrix[split:,:], labMatrix[split:]);
"""

"""
# Naive Bayes
print ''
print 'Naive Bayes'
nb = MultinomialNB();
nb.fit(newFeatureMatrix[:split,:], labMatrix[:split]);
print nb.score(newFeatureMatrix[split:,:], labMatrix[split:]);
"""

"""
# Neural Network
print ''
print 'Neural Network'
nn = MLPClassifier()
nn.fit(featureMatrix[:split, :], labMatrix[:split])
print nn.score(featureMatrix[split:,:],labMatrix[split:])

print ''
print '100 Decision tree'
treeList = [];
(row, col) = featureMatrix.shape;
featureMatrix = featureMatrix[:(row/2),:];
labMatrix = labMatrix[:(row/2)];
(row, col) = featureMatrix.shape
split = int(row*0.8);
trainset = featureMatrix[:split,:];
trainlab = labMatrix[:split];
testset = featureMatrix[split:,:];
testlab = labMatrix[split:];
(row, col) = trainset.shape;
for i in range(100):
	tree = DecisionTreeClassifier();
	trainIdx = [];
	while len(trainIdx)<row/10:
		rand = random.randint(0,row-1);
		if not rand in trainIdx:
			trainIdx.append(rand);

	train = trainset[trainIdx,:]
	lab = trainlab[trainIdx];
	tree.fit(trainset[trainIdx,:], trainlab[trainIdx]);
	treeList.append(tree);
	print len(treeList)

newTrainMatrix = [];
newTestMatrix = []
for tree in treeList:
	newTrainMatrix.append(np.array(tree.predict(trainset)))
	newTestMatrix.append(np.array(tree.predict(testset)));

newTrainMatrix = np.matrix(newTrainMatrix).T;
newTestMatrix = np.matrix(newTestMatrix).T;
print newTrainMatrix.shape, trainlab.shape
print testset.shape, testlab.shape
finalTree = DecisionTreeClassifier();
finalTree.fit(newTrainMatrix, trainlab);
print finalTree.score(newTestMatrix, testlab);
"""


"""
batch_size = 10;
learning_rate = 0.1;
activation_function = 'tanh';
hidden_layer_width = 10;
domain = 'mnist';
net = NN.create_NN(col, domain, batch_size, learning_rate, activation_function, hidden_layer_width)
net.train(trainMatrix);
print 'accuracy ' + str(net.evaluate(testMatrix));

net2 = NN.create_NN(col2, domain, batch_size, learning_rate, activation_function, hidden_layer_width)
net2.train(trainMatrix2);
print 'accuracy ' + str(net2.evaluate(testMatrix2));
"""


print 'finished'

