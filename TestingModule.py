import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score,recall_score,precision_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
#get_ipython().magic(u'matplotlib inline')
from sklearn import preprocessing
import calc_array
import staticinfo
import bagofwords
import  WaveTransform as pyt 
import math

EntrpyCala = calc_array.calc_array()




def Feature_extraction(codebook,inputdata):
    sampleFeatures=[]
    inputdata=[float(dt) for dt in inputdata]
    stainfo = staticinfo.Stats(inputdata)
    #Global features
    sampleFeatures.append(len(inputdata))
    sampleFeatures.append(math.sqrt(len(inputdata)))
    sampleFeatures.append(stainfo.avg())
    sampleFeatures.append(stainfo.stdev())
    sampleFeatures.append(stainfo.max())
    k=[dataa for dataa in inputdata if dataa>7.0]
    sampleFeatures.append(len([dataa for dataa in inputdata if dataa>7.0])/float(len(inputdata)))
    sampleFeatures.append(len([dataa for dataa in inputdata if dataa==0])/float(len(inputdata)))
    aEnergySpectum,dEnergySpectum=pyt.calc_energyspectrumrow(inputdata,'haar')
    for i in range(20):
        sampleFeatures.append(dEnergySpectum[i])
    #BOW features
    segment = 6
    histgram=bagofwords.representdata(inputdata,2,segment,codebook)
    if len(histgram)<=0:
        return 0
    for i in range(len(histgram)):
        sampleFeatures.append(histgram[i])
    return sampleFeatures





columns = ['EntropyLength','sqLength','mean','stdev','maxvalue','maxpecent','zeropecent']
for k in range(0,20):
    columns.append('Spream_%d' % k)
for j in range(0,250):
    columns.append("histgram_%d" % j)


scaler = joblib.load("scaler")
clf = joblib.load("RFClassifier.m")

codebook =np.loadtxt(open("center_6_250.csv","rb"),delimiter=",",skiprows=0)  
while True:
    features=[]
    filePath = input("input the target file path(use exit to quit out):")
    if filePath.strip()=="exit":
        break
    fdata = open(filePath,'rb').read()
    ETS=EntrpyCala.generate(filePath)
    #ETS= EntrpyCala.generate(fdata)


    features = Feature_extraction(codebook,ETS)
    if features==0:
        print("Error hanpend")
    features = tuple(features)
    Df = []
    Df.append(features)
    df2 =pd.DataFrame(Df,columns=columns)
    df2=pd.get_dummies(df2)
    X_test_std =scaler.transform(df2)
    flag = clf.predict(X_test_std)
    if flag[0]==1:
        print("malicious")
    elif flag[0]==0:
        print("clean")


    
    