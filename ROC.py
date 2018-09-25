# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 15:03:37 2018

@author: john
"""

import numpy as np
import pandas as pd
from time import time
from IPython.display import display
import visuals as vs
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import xgboost as xgb
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import roc_curve, auc 
from sklearn.metrics import classification_report  
import xgboost as xgb
# TODO：导入'GridSearchCV', 'make_scorer'和其他一些需要的库
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree, svm, ensemble
from sklearn.linear_model import SGDClassifier 
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import fbeta_score, accuracy_score,recall_score,precision_score

from sklearn.base import clone
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.learning_curve import learning_curve
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cross_validation import train_test_split




data=pd.read_csv("trainsmple6_250.csv")
file_flag = data['flag']

features = data.drop('flag',axis=1)





features = pd.get_dummies(features)
feature_label = features.columns[0:]
X_train,X_test,y_train,y_test =train_test_split(features,file_flag,test_size=0.2,random_state=0)


X_train1 = X_train
X_train2 = X_train
X_train3 = X_train

X_test1 = X_test
X_test2 = X_test
X_test3 = X_test

D_g_train= X_train
D_g_test = X_test
B_g_train= X_train
B_g_test = X_test

def plot_roc(y,yhat,name):
    from sklearn.metrics import precision_recall_curve, roc_curve, auc,classification_report,confusion_matrix 
    false_positive_rate,true_positive_rate,threads = roc_curve(y,yhat)
    roc_auc = (false_positive_rate,true_positive_rate)
    plt.plot(false_positive_rate,true_positive_rate,lw=1,label='ROC fold %s(area=%0.2f)' % (name,roc_auc))
    #画对角线
    plt.plot([0,1],[0,1],'--',color=(0.6,0.6,0.6),label='Luck')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc='lower right')
    plt.show()

#print("======================")
#feature1包含staticFeature


DWTFe = "Spream_%d"
HISTFE = "histgram_%d"
for i in range(20):
    dis=DWTFe % i
    X_train1=X_train1.drop(dis,axis=1)
    X_test1=X_test1.drop(dis,axis=1)
for i in range(140):
    dis = HISTFE % i
    X_train1=X_train1.drop(dis,axis=1)
    X_test1=X_test1.drop(dis,axis=1)


#X_train1 = X_train1.drop('EntropyLength',axis=1)
#X_test1 = X_test1.drop('EntropyLength',axis=1)
#X_train1 = X_train1.drop('sqLength',axis=1)
#X_test1 = X_test1.drop('sqLength',axis=1)
#X_train1,X_test1,y_train1,y_test1 =train_test_split(features1,file_flag,test_size=0.3,random_state=0)
#feature2包含 wave_decom
X_train2 = X_train2.drop('EntropyLength',axis=1)
X_train2 = X_train2.drop('sqLength',axis=1)
X_train2 = X_train2.drop('mean',axis=1)
X_train2 = X_train2.drop('stdev',axis=1)
X_train2 = X_train2.drop('maxvalue',axis=1)
X_train2 = X_train2.drop('maxpecent',axis=1)
X_train2 = X_train2.drop('zeropecent',axis=1)



for i in range(140):
    dis = HISTFE % i
    X_train2 = X_train2.drop(dis,axis=1)


X_test2 = X_test2.drop('EntropyLength',axis=1)
X_test2 = X_test2.drop('sqLength',axis=1)
X_test2 = X_test2.drop('mean',axis=1)
X_test2 = X_test2.drop('stdev',axis=1)
X_test2 = X_test2.drop('maxvalue',axis=1)
X_test2 = X_test2.drop('maxpecent',axis=1)
X_test2 = X_test2.drop('zeropecent',axis=1)
#X_test2 = X_test2.drop('fileType_docx',axis=1)
#X_test2 = X_test2.drop('fileType_pdf',axis=1)
#X_test2 = X_test2.drop('fileType_rtf',axis=1)
#X_test2 = X_test2.drop('fileType_ole',axis=1)


for i in range(140):
    dis = HISTFE % i
    X_test2 = X_test2.drop(dis,axis=1)




for i in range(20):
    dis=DWTFe % i
    X_test3=X_test3.drop(dis,axis=1)

X_test3 = X_test3.drop('EntropyLength',axis=1)
X_test3 = X_test3.drop('sqLength',axis=1)
X_test3 = X_test3.drop('mean',axis=1)
X_test3 = X_test3.drop('stdev',axis=1)
X_test3 = X_test3.drop('maxvalue',axis=1)
X_test3 = X_test3.drop('maxpecent',axis=1)
X_test3 = X_test3.drop('zeropecent',axis=1)
#X_test3 = X_test3.drop('fileType_docx',axis=1)
#X_test3 = X_test3.drop('fileType_pdf',axis=1)
#X_test3 = X_test3.drop('fileType_rtf',axis=1)
#X_test3 = X_test3.drop('fileType_ole',axis=1)


for i in range(20):
    dis=DWTFe % i
    X_train3=X_train3.drop(dis,axis=1)

HISTFE = "histgram_%d"   
'''
for i in range(140):
    dis = HISTFE % (i)
    X_train3=X_train3.drop(dis,axis=1)
    X_test3=X_test3.drop(dis,axis=1)
'''
X_train3 = X_train3.drop('EntropyLength',axis=1)
X_train3 = X_train3.drop('sqLength',axis=1)
X_train3 = X_train3.drop('mean',axis=1)
X_train3 = X_train3.drop('stdev',axis=1)
X_train3 = X_train3.drop('maxvalue',axis=1)
X_train3 = X_train3.drop('maxpecent',axis=1)
X_train3 = X_train3.drop('zeropecent',axis=1)

print("")

print(X_train3)
print("***********************************")
print(X_test3)


scaler1= MinMaxScaler()
scaler2= MinMaxScaler()
scaler3= MinMaxScaler()
scaler4= MinMaxScaler()
scaler5= MinMaxScaler()
scaler6= MinMaxScaler()
    
X_train =scaler1.fit_transform(X_train)
X_test  =scaler1.transform(X_test)
    

print(X_train3)


clfA = ensemble.RandomForestClassifier()

clfA.fit(X_train, y_train)
importances = clfA.feature_importances_

clfA1 = ensemble.RandomForestClassifier(n_estimators=500,max_depth=30)
y_predictions = (clfA1.fit(X_train, y_train)).predict(X_test)  
prob_predict_y_validation = clfA1.predict_proba(X_test)#给出带有概率值的结果，每个点所有label的概率和为1  
predictions_validation = prob_predict_y_validation[:, 1]  
fpr, tpr, _ = roc_curve(y_test, predictions_validation) 

tn,fp,fn,tp = confusion_matrix(y_test,y_predictions).ravel()

print("all Features ===========11111111111111111111111")
#print X_train1.head(n=1)
print("---000----")
print("TN = ",tn) 
print("TP = ",tp) 
print("FP = ",fp) 
print("FN = ",fn) 

pre=float(tp)/float(tp+fp)
recall=float(tp)/float(tp+fn)

print("FPR",float(fp)/float(fp+tn))
print("FNR",float(fn)/float(tp+fn))
print("TPR",recall)
print("pre",pre)

F1_score = (2*pre*recall)/(pre+recall)
#print("F1-score",F1_score)

 






roc_auc = auc(fpr, tpr)
#print roc_auc
clf1 = ensemble.RandomForestClassifier(n_estimators=500,max_depth=30)
y_predictions1 = (clf1.fit(X_train1, y_train)).predict(X_test1)  
prob_predict_y_validation1 = clf1.predict_proba(X_test1)#给出带有概率值的结果，每个点所有label的概率和为1  

tn,fp,fn,tp = confusion_matrix(y_test,y_predictions1).ravel()

print("Global Features ===========11111111111111111111111")
#print X_train1.head(n=1)
print("---000----")
print("TN = ",tn) 
print("TP = ",tp) 
print("FP = ",fp) 
print("FN = ",fn) 

pre=float(tp)/float(tp+fp)
recall=float(tp)/float(tp+fn)
print("FNR",float(fn)/float(tp+fn))
print("FPR",float(fp)/float(fp+tn))
print("TPR",recall)
print("pre",pre)

F1_score = (2*pre*recall)/(pre+recall)
#print "F1-score",F1_score

predictions_validation1 = prob_predict_y_validation1[:, 1]  
fpr1, tpr1, _ = roc_curve(y_test, predictions_validation1)  


roc_auc1 = auc(fpr1, tpr1)

clf = ensemble.RandomForestClassifier(n_estimators=500,max_depth=30)
y_predictions2 = (clf.fit(X_train2, y_train)).predict(X_test2)  
prob_predict_y_validation2 = clf.predict_proba(X_test2)#给出带有概率值的结果，每个点所有label的概率和为1  
predictions_validation2 = prob_predict_y_validation2[:, 1]  
fpr2, tpr2, _ = roc_curve(y_test, predictions_validation2)  

tn,fp,fn,tp = confusion_matrix(y_test,y_predictions2).ravel()

print("DWT Features ===========11111111111111111111111")
#print X_train1.head(n=1)
print("---000----")
print("TN = ",tn) 
print("TP = ",tp) 
print("FP = ",fp) 
print("FN = ",fn) 

pre=float(tp)/float(tp+fp)
recall=float(tp)/float(tp+fn)
print("FNR",float(fn)/float(tp+fn))
print("FPR",float(fp)/float(fp+tn))
print("TPR",recall)
print("pre",pre)

F1_score = (2*pre*recall)/(pre+recall)
#print "F1-score",F1_score




roc_auc2 = auc(fpr2, tpr2)

clf = ensemble.RandomForestClassifier(n_estimators=500,max_depth=30)
y_predictions3 = (clf.fit(X_train3, y_train)).predict(X_test3)  
prob_predict_y_validation3 = clf.predict_proba(X_test3)#给出带有概率值的结果，每个点所有label的概率和为1  
predictions_validation3 = prob_predict_y_validation3[:, 1]  
fpr3, tpr3, _ = roc_curve(y_test, predictions_validation3)  


tn,fp,fn,tp = confusion_matrix(y_test,y_predictions3).ravel()

print("BOW Features ===========11111111111111111111111")
#print X_train1.head(n=1)
print("---000----")
print("TN = ",tn) 
print("TP = ",tp) 
print("FP = ",fp) 
print("FN = ",fn) 

pre=float(tp)/float(tp+fp)
recall=float(tp)/float(tp+fn)
print("FNR",float(fn)/float(tp+fn))
print("FPR",float(fp)/float(fp+tn))
print("TPR",recall)
print("pre",pre)

F1_score = (2*pre*recall)/(pre+recall)
#print "F1-score",F1_score



roc_auc3 = auc(fpr3, tpr3)

#predict_prob_y = clf.predict_proba(test_x)#基于SVM对验证集做出预测，prodict_prob_y 为预测的概率


HISTFE = "histgram_%d"

for i in range(140):
    dis = HISTFE % i
    D_g_train=D_g_train.drop(dis,axis=1)
    D_g_test=D_g_test.drop(dis,axis=1)


#B_g_train= X_train
#B_g_test = X_test
HISTFE = "histgram_%d"
for i in range(20):
    dis=DWTFe % i
    B_g_train=B_g_train.drop(dis,axis=1)
    B_g_test=B_g_test.drop(dis,axis=1)

#print "=========================="
#print B_g_train.head(n=1)
    
B_g_train =scaler5.fit_transform(B_g_train)
B_g_train  =scaler5.transform(B_g_train)

D_g_train =scaler6.fit_transform(D_g_train)
D_g_test  =scaler6.transform(D_g_test)

clf = ensemble.RandomForestClassifier(n_estimators=500)
y_predictions3 = (clf.fit(D_g_train, y_train)).predict(D_g_test)  
prob_predict_y_validation3 = clf.predict_proba(D_g_test)#给出带有概率值的结果，每个点所有label的概率和为1  
predictions_validation3 = prob_predict_y_validation3[:, 1]  
fprDG, tprDG, _ = roc_curve(y_test, predictions_validation3)  
roc_aucDG = auc(fprDG, tprDG)

tn,fp,fn,tp = confusion_matrix(y_test,y_predictions3).ravel()
print("DWT+Global ===========11111111111111111111111")
print(X_train1.head(n=1))
print("---000----")
print("TN = ",tn) 
print("TP = ",tp) 
print("FP = ",fp) 
print("FN = ",fn) 
pre=float(tp)/float(tp+fp)
recall=float(tp)/float(tp+fn)
print("FNR",float(fn)/float(tp+fn))
print("FPR",float(fp)/float(fp+tn))
print("TPR",recall)
print("pre",pre)
F1_score = (2*pre*recall)/(pre+recall)
#print "F1-score",F1_score




clf = ensemble.RandomForestClassifier()
y_predictions3 = (clf.fit(B_g_train, y_train)).predict(B_g_test)  
prob_predict_y_validation3 = clf.predict_proba(B_g_test)#给出带有概率值的结果，每个点所有label的概率和为1  
predictions_validation3 = prob_predict_y_validation3[:, 1]  
fprBG, tprBG, _ = roc_curve(y_test, predictions_validation3) 
#print "We are here ing===================="
roc_aucBG=auc(fprBG, tprBG)
#print roc_aucDG

roc_auc = auc(fpr, tpr)

fig = plt.figure(figsize = (6.0,6))
plt.title('ROC Validation')  

plt.plot(fpr1*100,tpr1*100,'g-s',label='GlobalFeautes,AUC=%0.3f' % roc_auc1)


print("Global Features======================")
#print(fpr1*100)

#print(tpr1*100)
plt.plot(fpr2*100,tpr2*100,'r-o',label='DWT,AUC=%0.3f' % roc_auc2)
print("DWT Features======================")
#print(fpr2*100)

#print(tpr2*100)

plt.plot(fpr3*100,tpr3*100,'k-d',label='BOW,AUC=%0.3f' % roc_auc3)

print("BOW AUC Features======================")
#print(fpr3*100)

#print(tpr3*100)

#plt.plot(fpr3*100,tpr3*100,'k-d',label='BOW,AUC=%0.3f' % 0.9684)
plt.plot(fprDG*100,tprDG*100,'y-H',label='DWT+Global,AUC=%0.3f' % roc_aucDG)

print("DWT+Global Features======================")
#print(fprDG*100)

print(tprDG*100)
plt.plot(fpr*100,tpr*100,'b-*',label='AllFeatures,AUC=%0.3f' % roc_auc)
print("AllFeature Features======================")
#print(fpr*100)

#print(tpr*100)

#plt.plot(fprBG*100,tprBG*100,'c-h',label='BOW+G,AUC=%0.2f' % roc_auc3)
plt.legend(loc='lower right')  
#plt.plot([0, 1], [0, 1], 'r--')  
plt.xlim([0, 100])  
plt.ylim([80, 100])  
plt.ylabel('True Positive Rate(%)')  
plt.xlabel('False Positive Rate(%)')  
plt.grid()
plt.savefig("ROC.eps")
plt.show()

plt.close()








