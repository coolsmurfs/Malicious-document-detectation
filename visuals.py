# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
from IPython import get_ipython

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score,accuracy_score


def distribution(data,transformed=False):
    '''
    Visualization code for dispalying skewed distributions of features
    '''
    #Create figure
    fig = pl.figure(figsize=(11,5))
    #Skewed feature ploting
    for i,feature,in enumerate(['size']):
        ax = fig.add_subplot(1,2,i+1)
        ax.hist(data[feature],bins=25,color='#00A0A0')
        ax.set_title("'%s' Feature Distribution" % (feature),fontsize=14)
        ax.set_label("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0,2000))
        ax.set_yticks([0,500,1000,1500,2000])
        ax.set_yticklabels([0,500,1000,1500,">2000"])
    #plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Census Data Features", \
            fontsize = 16, y = 1.03)
    else:
       fig.suptitle("Skewed Distributions of Continuous Census Data Features", \
            fontsize = 16, y = 1.03)
    fig.tight_layout()
    fig.show()
     
     

    
results = {'DecisionTreeClassifier': [{'FNR': 12.966476913345984, 'PRE': 87.0886075949367, 'train_time': 0.1500084400177002, 'TPR': 87.03352308665403, 'F1_score': 87.06105662764948, 'FPR': 13.48314606741573}, {'FNR': 13.851992409867172, 'PRE': 87.58842443729903, 'train_time': 0.28601646423339844, 'TPR': 86.14800759013282, 'F1_score': 86.8622448979592, 'FPR': 12.756113681427628}, {'FNR': 10.62618595825427, 'PRE': 88.53383458646617, 'train_time': 0.43602490425109863, 'TPR': 89.37381404174573, 'F1_score': 88.95184135977338, 'FPR': 12.095175148711169}, {'FNR': 9.550917141049968, 'PRE': 88.54489164086688, 'train_time': 0.7920453548431396, 'TPR': 90.44908285895004, 'F1_score': 89.48685857321652, 'FPR': 12.227362855254462}], 'RandomForestClassifier': [{'FNR': 12.966476913345984, 'PRE': 91.85580774365822, 'train_time': 0.10100579261779785, 'TPR': 87.03352308665403, 'F1_score': 89.37966872361157, 'FPR': 8.06345009914078}, {'FNR': 10.879190385831754, 'PRE': 92.8806855636124, 'train_time': 0.17200970649719238, 'TPR': 89.12080961416825, 'F1_score': 90.96191091026469, 'FPR': 7.13813615333774}, {'FNR': 9.108159392789373, 'PRE': 93.67666232073013, 'train_time': 0.28601646423339844, 'TPR': 90.89184060721063, 'F1_score': 92.26324237560193, 'FPR': 6.411103767349637}, {'FNR': 7.590132827324478, 'PRE': 95.17915309446255, 'train_time': 0.36802101135253906, 'TPR': 92.40986717267553, 'F1_score': 93.77406931964056, 'FPR': 4.890945142101785}], 'LogisticRegression': [{'FNR': 19.671094244149273, 'PRE': 77.81862745098039, 'train_time': 0.07900476455688477, 'TPR': 80.32890575585073, 'F1_score': 79.0538437597261, 'FPR': 23.92597488433576}, {'FNR': 20.240354206198607, 'PRE': 78.8125, 'train_time': 0.28401637077331543, 'TPR': 79.7596457938014, 'F1_score': 79.28324426281043, 'FPR': 22.405816259087903}, {'FNR': 19.671094244149273, 'PRE': 78.735275883447, 'train_time': 0.5130293369293213, 'TPR': 80.32890575585073, 'F1_score': 79.52410770194113, 'FPR': 22.670191672174486}, {'FNR': 19.35483870967742, 'PRE': 80.18867924528303, 'train_time': 0.8610491752624512, 'TPR': 80.64516129032258, 'F1_score': 80.4162724692526, 'FPR': 20.819563780568405}], 'XGBClassifier': [{'FNR': 10.815939278937384, 'PRE': 92.45901639344261, 'train_time': 1.3960797786712646, 'TPR': 89.18406072106262, 'F1_score': 90.7920154539601, 'FPR': 7.600793126239259}, {'FNR': 10.373181530676787, 'PRE': 93.65499008592201, 'train_time': 2.289130926132202, 'TPR': 89.62681846932321, 'F1_score': 91.59663865546217, 'FPR': 6.345009914077991}, {'FNR': 9.867172675521822, 'PRE': 92.7734375, 'train_time': 3.0651752948760986, 'TPR': 90.13282732447819, 'F1_score': 91.43407122232917, 'FPR': 7.336417713152676}, {'FNR': 10.056925996204933, 'PRE': 94.04761904761905, 'train_time': 4.154237747192383, 'TPR': 89.94307400379506, 'F1_score': 91.94956353055288, 'FPR': 5.948446794448117}], 'SVC': [{'FNR': 29.98102466793169, 'PRE': 70.59948979591837, 'train_time': 2.416138172149658, 'TPR': 70.01897533206831, 'F1_score': 70.3080342966021, 'FPR': 30.46926635822869}, {'FNR': 23.719165085388994, 'PRE': 70.94117647058825, 'train_time': 9.739557027816772, 'TPR': 76.280834914611, 'F1_score': 73.5141725083816, 'FPR': 32.650363516192996}, {'FNR': 19.038583175205567, 'PRE': 69.11447084233261, 'train_time': 19.480114221572876, 'TPR': 80.96141682479443, 'F1_score': 74.57034663559568, 'FPR': 37.805684071381364}, {'FNR': 17.52055660974067, 'PRE': 69.2144373673036, 'train_time': 38.87222337722778, 'TPR': 82.47944339025933, 'F1_score': 75.26695526695526, 'FPR': 38.33443489755452}]}

def evaluate1(results):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """
  
    # Create figure
    fig, ax = pl.subplots(2, 3, figsize = (11,7))
    #fig, ax = pl.subplots(2, 3)
    # Constants
    bar_width = 0.3
    colors = ['#A00000','#00A0A0','#00A000']
              
    colors =['b','g','y','k','r','c']
    labels=['DecisionTree','svm','AdaBoost','RandomForest','GradientBoost','LogisticRegress']
    ylabel=['Time (in seconds)','TPR score(%)','Precision score(%)',"FPR score(%)",'FNR score(%)','F1-score(%)']
    Title =["Model Training","TPR score on Testing Set","Precision Score on Testing test",
            "FPR score on Testing Set","FNR score on Testing Set","F-score on Testing Set"]
    
    #print results
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        #for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
        for j, metric in enumerate(['train_time', 'TPR', 'PRE', 'FPR', 'FNR', 'F1_score']):
            data =[results[learner][0][metric],results[learner][1][metric],results[learner][2][metric],results[learner][3][metric]]
         
            index = int(j/3)
            if learner=="XGBClassifier":
                #ax[j/3, j%3].plot([0.0, 0.857, 1.71,3.0],data,'%ss--' % colors[k],label="$%s$" % learner)
                ax[index, j%3].plot([0.0, 0.857, 1.71,3.0],data,'%ss-' % colors[k],label="SGB")
            elif learner =="LogisticRegression":
                #ax[j/3, j%3].plot([0.0, 0.857, 1.71,3.0],data,'%s*--' % colors[k],label="$%s$" % learner)
                ax[index, j%3].plot([0.0, 0.857, 1.71,3.0],data,'%s*-' % colors[k],label="LR")
            elif learner =="DecisionTreeClassifier":
                #ax[j/3, j%3].plot([0.0, 0.857, 1.71,3.0],data,'%so--' % colors[k],label="$%s$" % learner)
                ax[index, j%3].plot([0.0, 0.857, 1.71,3.0],data,'%so-' % colors[k],label="DT")
            elif learner =="SVC":
               # ax[j/3, j%3].plot([0.0, 0.857, 1.71,3.0],data,'%sv--' % colors[k],label="$%s$" % learner)
               ax[index, j%3].plot([0.0, 0.857, 1.71,3.0],data,'%sv-' % colors[k],label="SVC")
            elif learner =="RandomForestClassifier":
                #ax[j/3, j%3].plot([0.0, 0.857, 1.71,3.0],data,'%s>--' % colors[k],label="$%s$" % learner)
                ax[index, j%3].plot([0.0, 0.857, 1.71,3.0],data,'%s>-' % colors[k],label="RF")
                
            ax[index, j%3].legend()
            ax[index, j%3].set_xticks([0.0, 0.857, 1.71,3.0])
            ax[index, j%3].set_xticklabels(["30%", "50%","70%", "100%"])
            #ax[j/3, j%3].set_xlabel("Training Set Size")
            ax[index, j%3].set_xlim((-0.1, 3.1))
                #ax[j/3, j%3].set_xlim((-0.1, 3.3))
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("TPR score(%)")
    #ax[0, 1].set_ylim((0,1))
    ax[0, 2].set_ylabel("Precision score(%)")
    #ax[0, 2].set_ylim((0,1))
    ax[1, 0].set_ylabel("FPR score(%)")
    ax[1, 1].set_ylabel("FNR score(%)")
    #ax[1, 1].set_ylim((0,1))
    ax[1, 2].set_ylabel("F1-score(%)")
    #ax[1, 2].set_ylim((0,1))
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("TPR score on Testing set")
    ax[0, 2].set_title("Precision Score on Testing set")
    ax[1, 0].set_title("FPR score on Testing set")
    ax[1, 1].set_title("FNR score on Testing set")
    ax[1, 2].set_title("F-score on Testing set")
    
    # Add horizontal lines for naive predictors
    #ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    #ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    #ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    #ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    
    # Set y-limits for score panels
    #ax[0, 1].set_ylim((0, 1))
    #ax[0, 2].set_ylim((0, 1))
    #ax[1, 1].set_ylim((0, 1))
    #ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    #for i, learner in enumerate(results.keys()):
    #    patches.append(mpatches.Patch(color = colors[i], label = learner))
    #pl.legend(handles = patches, bbox_to_anchor = (-.80, 2.63), \
    #           loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 10)
    
    # Aesthetics
    #pl.suptitle("Performance Metrics for Five Supervised Learning Models", fontsize = 14, y = 1.10)
    #pl.tight_layout()
    pl.savefig("classifier.eps")
    pl.show()

     
def feature_plot(importances, X_train, y_train):
    
    # Display the five most important features
    print("hahahah=====")
    print(importances)
    #importances = sorted(importances)
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:]]
    print(columns)
    values = importances[indices][:]

    # Creat the plot
    fig = pl.figure(figsize = (18,7))
    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize = 14)
    pl.bar(np.arange(len(values)), values, width = 0.2, align="center", color = '#00A000', \
          label = "Feature Weight")
    #pl.bar(np.arange(len(values)) - 0.2, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
    #      label = "Cumulative Feature Weight")
    pl.xticks(np.arange(len(values)), columns)
    pl.xlim((-0.5, 20))
    pl.ylabel("Weight", fontsize = 12)
    pl.xlabel("Feature", fontsize = 12)
    
    pl.legend(loc = 'upper center')
    pl.tight_layout()
    pl.savefig("123.png")
    pl.show()  
    pl.close()
    
    
        
                


