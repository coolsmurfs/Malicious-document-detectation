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
from IPython.display import display
import visuals as vs
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
#import xgboost as xgb
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_curve, auc 
from sklearn.metrics import classification_report  

# TODO：导入'GridSearchCV', 'make_scorer'和其他一些需要的库
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree, svm, ensemble
from sklearn.linear_model import SGDClassifier 


from sklearn.base import clone
from itertools import combinations
import numpy as np
#from sklearn.cross_validation import train_test_split
import os
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


from sklearn.metrics import fbeta_score, accuracy_score,recall_score,precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


averageAA={}
fig = plt.figure(0)
for k in range(10):
    codebooksize={}
    KbookSize =[]
    for root,dirs,files in os.walk(os.path.join(os.getcwd(),'codebook_size')):
        for filename in files:
            result = {}
            if filename.endswith("docx"):
                continue
            booksize = int(filename.split('_')[1][:-4])
            print(booksize)
            #break
            KbookSize.append(booksize)
            filepath = os.path.join(root,filename)
            data=pd.read_csv(filepath)
            
            file_flag = data['flag']
            features = data.drop('flag',axis=1)
            X_train,X_test,y_train,y_test =train_test_split(features,file_flag,test_size=0.3,random_state=0)
            scaler= MinMaxScaler()
            #print "here we stop"
            X_train_std =scaler.fit_transform(X_train)
            X_test_std =scaler.transform(X_test)
            clf = ensemble.RandomForestClassifier()
            clf.fit(X_train_std, y_train)
            prob_predict_y_validation1 = clf.predict_proba(X_test_std)#基于SVM对验证集做出预测，prodict_prob_y 为预测的概率
            predictions_validation1 = prob_predict_y_validation1[:, 1]  
            fpr1, tpr1, _ = roc_curve(y_test, predictions_validation1)  
    
            #print "{}}}}}}}}}}}}}}}}}}"
            roc_auc1 = auc(fpr1, tpr1)
            
            predict=clf.predict(X_test_std)
            accuaracy = accuracy_score(y_test,predict)
            tn,fp,fn,tp = confusion_matrix(y_test,predict).ravel()

            result["ROC_AUC"]=roc_auc1
            
            codebooksize[booksize]=result
    
    averageAA[k]=codebooksize

kleng = len(codebooksize)
dataValue={}
for i in codebooksize.keys():
    dataValue[i]=[]
#print average     
 
for key,values in enumerate(averageAA.keys()):
    
    data = averageAA[key]
    print("}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print(data)
    #print "{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}"
    #print data
    for i in data.keys():
        dataValue[i].append(data[i]['ROC_AUC'])
data={}    
print(codebooksize)
for key in dataValue.keys():
    data[key]= float(sum(dataValue[key]))/len(dataValue[key])
#print codebooksize
KbookSize.sort()
#x=KbookSize
x = KbookSize#


roc_auc = [data[i] for i in x]  
#print "jiuzheng========"
print("==========================")
print(x)
print(roc_auc)
print("###########################")

fig = plt.figure()
p3=plt.plot(x, roc_auc,'k*-',label="ROC_auc") 
#p4=plt.plot(x, fpr, 'kD-',label="FPR") 
#l1 = plt.legend([p1, p2,p3], ["precision", "recall", "accuracy"], loc='lower right') 
legend = plt.legend(loc='best', shadow=True,)
plt.xlabel("codebook size")
plt.ylabel("AUC value")
plt.grid(True)
#plt.title('Classiferi*RandomForest',fontsize=16)
plt.savefig("codebooks.eps")
plt.show()
print(codebooksize)





