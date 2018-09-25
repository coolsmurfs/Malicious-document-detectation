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



#print [samples_10, samples_30, samples_50,samples_70,samples_100]



#data=pd.read_csv("trainsmpl0e08_80.csv")
#print("hahahh==============")
#GFlag = data['flag']


averageAA={}
fig = plt.figure(0)
for k in range(10):
    codebooksize={}
    KbookSize =[]
    for root,dirs,files in os.walk(os.path.join(os.getcwd(),'TrainDataSet')):
        for filename in files:
            result = {}
            #booksize = int(filename.split('_')[0][-2:])
            booksize = int(filename.split('_')[1][:-4])
            print(booksize)
            #break
            KbookSize.append(booksize)
            filepath = os.path.join(root,filename)
            data=pd.read_csv(filepath)
            try:
                file_flag = data['flag']
                features = data.drop('flag',axis=1)
            except:
                file_flag=GFlag
                features = data
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
    #print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    #print averageAA

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
#print "#############################"
#print data
#print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
#pre = [codebooksize[i]['PRE'] for i in x]      
#recall = [codebooksize[i]['TPR'] for i in x]  

roc_auc = [data[i] for i in x]  
#print "jiuzheng========"
print("==========================")
print(x)
print(roc_auc)
print("###########################")
#fpr = [codebooksize[i]['FPR'] for i in x] 
#plt.scatter(x,pre,marker='o-',c='',edgecolors='blue')
#p1=plt.plot(x, pre, 'bs-',label="PRE") 
#plt.scatter(x,recall,marker='*-',c='',edgecolors='red')
#p2=plt.plot(x, recall, 'r*-',label="TPR") 
#plt.scatter(x,accruracy,marker='D-',c='',edgecolors='green')
#fig = plt.figure(figsize=(5,4))
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
'''
plotacc = {}
for n_estimators in range(100,3000,100):
    
    clf = ensemble.RandomForestClassifier(n_estimators)
    #clf = svm.SVC()
    clf.fit(X_train_std, y_train)
    predict=clf.predict(X_test)
    accscore = accuracy_score(predict,y_test)
    
    joblib.dump(clf,'RandomForest.m')
    #clf =xgb.XGBClassifier(n_estimators=1000)
    #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2,1e-3, 1e-4],
    #                     'C': [1, 10, 100, 1000]}]
    scores = ['precision']
    
    y_predictions = (clf.fit(X_train_std, y_train)).predict(X_test_std)  
    
    #y_predictions = lr.predict(X_test_std)  
    
    prob_predict_y_validation = clf.predict_proba(X_train_std)#给出带有概率值的结果，每个点所有label的概率和为1  
    predictions_validation = prob_predict_y_validation[:, 1]  
    fpr, tpr, _ = roc_curve(y_train, predictions_validation)  
        #  
    print len(fpr)
    print len(tpr)
    roc_auc = auc(fpr, tpr)  
    plt.title('ROC Validation')  
    plt.plot(fpr,tpr)
    #plt.plot([0,0.1,0.2,0.3,0.4,0.5,0.9], tpr[0.2,0.7,0.8,0.9,0.9,0.9,0.9], 'b--', label='AUC = %0.2f' % roc_auc)  
    plt.legend(loc='lower right')  
    
    plt.ylabel('True Positive Rate')  
    plt.xlabel('False Positive Rate')  
    plt.show()

    from sklearn.metrics import accuracy_score
    accuaracy = accuracy_score(y_test,y_predictions)
    tn,fp,fn,tp = confusion_matrix(y_test,y_predictions).ravel()
    
    print X_train.shape
    
    print "TN = ",tn 
    print "TP = ",tp 
    print "FP = ",fp 
    print "FN = ",fn 
    print "precision=",float(tp)/float(tp+fp)
    print "recall=",float(tp)/float(tp+fn)
    print "accuracy=",(float(tp)+float(tn))/float(fp+tp+tn+fn)
    print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, y_predictions, beta = 0.5))
    plotacc[n_estimators]=(float(tp)+float(tn))
X=[]
Y=[]
for i in plotacc.keys():
    X.append(i)
    Y.append(plotacc[i]*100)
    
fig = plt.figure()
plt.plot(X,Y)
plt.show()
plt.close()
'''
#绘制ROC曲线




'''
from sklearn.pipeline import Pipeline
from sklearn.learning_curve import learning_curve

#pipe_lr = Pipeline([('clf',ensemble.RandomForestClassifier(n_estimators=10))])

pipe_lr = Pipeline([('clf',svm.SVC())])

train_size,train_scores,test_score = learning_curve(estimator=pipe_lr,X=X_train,y=y_train,train_sizes=np.linspace(0.1,1.0,10),
                                                   cv=10,n_jobs=1)


print len(train_size)
train_mean = np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean = np.mean(test_score,axis=1)
test_std = np.std(test_score,axis=1)
plt.plot(train_size,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(train_size,train_mean+train_std,train_mean-train_std,
                alpha=0.15,color='blue')
plt.plot(train_size,test_mean,color='red',linestyle='--',marker='s',markersize=5,label='validation accuracy')
plt.fill_between(train_size,test_mean+test_std,test_mean-test_std,alpha=0.15,color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.5,1.0])

plt.savefig('Learn.eps')
plt.show()
plt.close()

'''






# TODO：创建你希望调节的参数列表
#print "=============================================="
#print(max(results, key=lambda x: x[2]))
#print results

#导入一个有‘features_importances’的监督学习模型
# TODO：导入一个有'feature_importances_'的监督学习模型
'''
from sklearn.ensemble import RandomForestClassifier

# TODO：在训练集上训练一个监督学习模型
#model = RandomForestClassifier(n_estimators=1000)
model = svm.SVC(C=0.1)
#model =xgb.XGBClassifier(n_estimators=10)
#model = LogisticRegression(C=1000.0,random_state=0)


display(X_train.head(n=1))
model.fit(X_train,y_train)


testData = data=pd.read_csv("TestAllrow.csv")

testFlag = testData['flag']
testFeatures = testData.drop('flag',axis=1)

scaler = MinMaxScaler()
numerical =['EntropyLength','sqLength','mean','stdev','maxvalue','maxpecent','zeropecent','aenergy_level_1','aenergy_level_2','aenergy_level_3','aenergy_level_4','aenergy_level_5','aenergy_level_6','aenergy_level_7','aenergy_level_8','aenergy_level_9', 'aenergy_level_10','aenergy_level_11' ,'aenergy_level_12','aenergy_level_13','aenergy_level_14','aenergy_level_15','aenergy_level_16',
            "denergy_level_1","denergy_level_2","denergy_level_3","denergy_level_4","denergy_level_5","denergy_level_6","denergy_level_7","denergy_level_8","denergy_level_9", "denergy_level_10","denergy_level_11","denergy_level_12","denergy_level_13","denergy_level_14","denergy_level_15","denergy_level_16"]
#numerical.append()
#numerical =['EntropyLength','sqLength','mean','stdev','maxvalue','maxpecent','zeropecent','energy_level_1','energy_level_2','energy_level_3','energy_level_4','energy_level_5','energy_level_6','energy_level_7','energy_level_8','energy_level_9','energy_level_10','energy_level_11','energy_level_12','energy_level_13','energy_level_14','energy_level_15','energy_level_16']
#testFeatures[numerical]=scaler.fit_transform(testFeatures[numerical])

'''




