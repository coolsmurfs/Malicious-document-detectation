# -*- coding: utf-8 -*-
"""
Created on Sat Jul 01 20:02:48 2017

@author: john
"""

# coding: utf-8

# In[36]:

import numpy as np
import pandas as pd
from time import time
import visuals as vs
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import xgboost as xgb
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_curve, auc 
from sklearn.metrics import classification_report  

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree, svm, ensemble
from sklearn.linear_model import SGDClassifier 
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score,recall_score,precision_score
from SBS import *
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors 
#get_ipython().magic(u'matplotlib inline')
from sklearn import preprocessing
from sklearn.externals import joblib

def plot_roc(y,yhat,name):
    from sklearn.metrics import precision_recall_curve, roc_curve, auc,classification_report,confusion_matrix 
    false_positive_rate,true_positive_rate,threads = roc_curve(y,yhat)
    roc_auc = (false_positive_rate,true_positive_rate)
    plt.plot(false_positive_rate,true_positive_rate,lw=1,label='ROC fold %s(area=%0.2f)' % (name,roc_auc))
    plt.plot([0,1],[0,1],'--',color=(0.6,0.6,0.6),label='Luck')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc='lower right')
    plt.show()
    

# TODO
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
   
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
       
    '''
    
    results = {}
    from sklearn import metrics

    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    start = time() # training start time
    learner = learner.fit(X_train[: sample_size], y_train[: sample_size])
    end = time() # training end time
    
    # TODO：
    results['train_time'] = end - start
    start = time() # predict start time
    predictions_test = learner.predict(X_test)
    #
    end = time() # predict end time
    fpr, tpr, thresholds = roc_curve(y_test, predictions_test, pos_label=2)
    auc=metrics.auc(fpr, tpr)
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=1)
    
    
    tn,fp,fn,tp = confusion_matrix(y_test,predictions_test).ravel()
    results['TPR'] = float(tp)/float(tp+fn)*float(100)
    #print "==============================",float(tp)/float(fp+fn),"=========================="
    results['PRE'] = float(tp)/float(tp+fp)*float(100)
    results['FPR'] =float(fp)/float(fp+tn)*float(100) #accuracy_score(y_test, predictions_test)
    results['FNR'] = float(fn)/float(tp+fn)*float(100)
    results['F1_score'] = fbeta_score(y_test, predictions_test, beta=1)*float(100)
    return results




data=pd.read_csv("trainsmple6_250.csv")
flag = data['flag']
print("Here read the data")

#Total number
n_records = data.shape[0]
#malware
n_malware_files=data[data.flag==1].shape[0]
#Benign file
n_clean_files = data[data.flag==0].shape[0]

print("sample==========")
#print n_malware_files
print("=========================")
#Proportion
malware_percent = np.divide(n_malware_files,float(n_records))*100

file_flag = flag

features = data.drop('flag',axis=1)


#one-hot encoding
features = pd.get_dummies(features)
#print features.head(n=4)

X_train,X_test,y_train,y_test =train_test_split(features,file_flag,test_size=0.3,random_state=0)
#Minser-Scaers
scaler= MinMaxScaler()
X_train_std =scaler.fit_transform(X_train)
X_test_std =scaler.transform(X_test)

joblib.dump(scaler,"scaler")

# display split result
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))   

#Decison Tree
clf_A = tree.DecisionTreeClassifier()
#svm
clf_B = svm.SVC()
#logic
clf_D = ensemble.RandomForestClassifier()


clf_F = LogisticRegression() 
clf_G =  xgb.XGBClassifier()
clf_K =neighbors.KNeighborsClassifier()
samples_30 =int(X_train.shape[0]*0.3)
samples_50 = int(X_train.shape[0]*0.5)
samples_70 = int(X_train.shape[0]*0.7)
samples_100= int(X_train.shape[0]*1)
results = {}
StaticResults = {}
for indk in range(3):
    for clf in [clf_A,clf_B,clf_D,clf_F,clf_G]:
        clf_name = clf.__class__.__name__
        results[clf_name]=[]
        index=0
        for i,samples in enumerate([samples_30,samples_50,samples_70,samples_100]):
            results[clf_name].append(train_predict(clf,samples,X_train_std,y_train,X_test_std,y_test))
            index+=1
    StaticResults[indk]=results

TotalResults={}


singleItem = {}
#for keydata in StaticResults.keys():
learns = []
for i,klear in enumerate(StaticResults[0].keys()):
    learns.append(klear)
parameterInfom={}
for learn in learns:
    parameterInfom[learn]=[]
    for ration in range(len(StaticResults[0][learn])):
        KeyInfo={}
        for j, metric in enumerate(['train_time', 'TPR','PRE', 'FPR',  'FNR', 'F1_score']):
            rationInfo=[]
            for i,datainfo in enumerate(StaticResults.keys()):
                rationInfo.append(StaticResults[datainfo][learn][ration][metric])
            KeyInfo[metric]=float(sum(rationInfo))/len(rationInfo)
        parameterInfom[learn].append(KeyInfo)

print("results:#############################")

for key,value in parameterInfom.items():
	print("classifier:%s\n"%key)
	print(value)
print("######################")
    
#display the result
vs.evaluate1(parameterInfom)
   



#
'''
  Select the best configuration for classifiers
  First: N_estimator  
'''
X=[]
Y=[]

plotacc = {}
for n_estimatorsP in range(10,100,10):
    clf = ensemble.RandomForestClassifier(n_estimators=n_estimatorsP)
    clf.fit(X_train_std, y_train)
    y_predictions = (clf.fit(X_train_std, y_train)).predict(X_test_std)  
    prob_predict_y_validation = clf.predict_proba(X_test_std)# 
    predictions_validation = prob_predict_y_validation[:, 1]  
    fpr, tpr, _ = roc_curve(y_test, predictions_validation)  
    roc_auc = auc(fpr, tpr)  

    #accuaracy = accuracy_score(y_test,y_predictions)
    tn,fp,fn,tp = confusion_matrix(y_test,y_predictions).ravel()
    print("n_estimators:%d" % n_estimatorsP)
    print("TN = ",tn) 
    print("TP = ",tp) 
    print("FP = ",fp) 
    print("FN = ",fn) 
    print("precision=",float(tp)/float(tp+fp))
    print("recall=",float(tp)/float(tp+fn))
    print("accuracy=",(float(tp)+float(tn))/float(fp+tp+tn+fn))
    print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, y_predictions, beta = 0.5)))
    plotacc[n_estimatorsP]=(roc_auc)



for i in plotacc.keys():
    X.append(i)
print(X)
X.sort()
print(X)
for i in X:
    Y.append(plotacc[i]*100)


plotacc = {}
for n_estimatorsP in range(100,2000,100):
    clf = ensemble.RandomForestClassifier(n_estimators=n_estimatorsP)
    clf.fit(X_train_std, y_train)
    y_predictions = (clf.fit(X_train_std, y_train)).predict(X_test_std)  
    prob_predict_y_validation = clf.predict_proba(X_test_std)# 
    predictions_validation = prob_predict_y_validation[:, 1]  
    fpr, tpr, _ = roc_curve(y_test, predictions_validation)  
    roc_auc = auc(fpr, tpr)  

    #accuaracy = accuracy_score(y_test,y_predictions)
    tn,fp,fn,tp = confusion_matrix(y_test,y_predictions).ravel()
    print("n_estimators:%d" % n_estimatorsP)
    print("TN = ",tn) 
    print("TP = ",tp) 
    print("FP = ",fp) 
    print("FN = ",fn) 
    print("precision=",float(tp)/float(tp+fp))
    print("recall=",float(tp)/float(tp+fn))
    print("accuracy=",(float(tp)+float(tn))/float(fp+tp+tn+fn))
    print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, y_predictions, beta = 0.5)))
    plotacc[n_estimatorsP]=(roc_auc)

X1=[]
for i in plotacc.keys():
    X1.append(i)
print(X1)
X1.sort()
print(X)
for i in X1:
    Y.append(plotacc[i]*100)
X=X+X1   
print(X)
print(Y)
fig = plt.figure()
plt.plot(X,Y)
plt.show()



'''
  Select the best configuration for classifiers
  Second: Max_depth 
'''
plotacc = {}
n_estimatorsP = 500
for max_depthP in range(10,100,10):
    clf = ensemble.RandomForestClassifier(max_depth=max_depthP,n_estimators=n_estimatorsP)
    clf.fit(X_train_std, y_train)
    y_predictions = (clf.fit(X_train_std, y_train)).predict(X_test_std)  
    prob_predict_y_validation = clf.predict_proba(X_test_std)# 
    predictions_validation = prob_predict_y_validation[:, 1]  
    fpr, tpr, _ = roc_curve(y_test, predictions_validation)  
    roc_auc = auc(fpr, tpr)  

    #accuaracy = accuracy_score(y_test,y_predictions)
    tn,fp,fn,tp = confusion_matrix(y_test,y_predictions).ravel()
    print("max_depth:%d" % max_depthP)
    print("TN = ",tn) 
    print("TP = ",tp) 
    print("FP = ",fp) 
    print("FN = ",fn) 
    print("precision=",float(tp)/float(tp+fp))
    print("recall=",float(tp)/float(tp+fn))
    print("accuracy=",(float(tp)+float(tn))/float(fp+tp+tn+fn))
    print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, y_predictions, beta = 0.5)))
    plotacc[max_depthP]=(roc_auc)

X=[]
Y=[]
print(plotacc)
for i in plotacc.keys():
    X.append(i)
X.sort()
for i in X:
    Y.append(plotacc[i]*100)
   

    
print(X,Y)
fig = plt.figure()
plt.plot(X,Y)
plt.show()

#


'''
  Finally: We set the N_estimator with 500 and max_depth with 30
'''

clf = ensemble.RandomForestClassifier(n_estimators=500,max_depth=30)
clf.fit(X_train_std, y_train)
joblib.dump(clf,"RFClassifier.m")



