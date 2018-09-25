# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from time import time

import visuals as vs
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import xgboost as xgb

from sklearn.metrics import classification_report  

# TODO：导入'GridSearchCV', 'make_scorer'和其他一些需要的库
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree, svm, ensemble
from sklearn.linear_model import SGDClassifier 


from sklearn.base import clone
from itertools import combinations
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


from sklearn.metrics import fbeta_score, accuracy_score,recall_score,precision_score
class SBS():
    def __init__(self,estimator,k_features,scoring=accuracy_score,test_size=0.25,random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
    def fit(self,X,y):
        X_train,X_test,y_train,y_test= train_test_split(X,y,test_size = self.test_size,random_state = self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        print("we are here to see something")
        print(self.indices_)
        score = self._calc_score(X_train,y_train,X_test,y_test,self.indices_)
        self.scores_=[score]
        while dim>self.k_features:
            scores = []
            subsets=[]
            for p in combinations(self.indices_,r=dim-1):
                #print "Here is the p"
                #print len(p)
                score = self._calc_score(X_train,y_train,X_test,y_test,p)
                scores.append(score)
                subsets.append(p)
            #print "dim here:",dim
            best = np.argmax(scores)
            self.indices = subsets[best]
            self.subsets_.append(self.indices_)
            dim=dim-1
            self.scores_.append(scores[best])
        self.k_score = self.scores_[-1]
        return self
    def transform(self,X):
        return X[:,self.indices_]
    def _calc_score(self,X_train,y_train,X_test,y_test,indices):
        #$print "Here we run the shell============-============"
        indices = list(indices)
        #p#rint indices
        #print X_train.iloc[:,indices]
        #print type(indices)
        #print "}}}}}}}}}}}}}}}}}}}}}{{{{{{{{{{{{{{{{{{{{{{"
        #print X_train.iloc[:,list(indices)]
        #print "================="
        self.estimator.fit(X_train.iloc[:,indices],y_train)
        #print "We are here do something"
        y_pred = self.estimator.predict(X_test.iloc[:,indices])
        score = self.scoring(y_test,y_pred)
        #print score
        return score