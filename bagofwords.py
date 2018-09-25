# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from numpy import *  
import time  
import random
import pandas as pd
from WaveTransform import *
from numpy import *  
import time  



# calculate Euclidean distance  
def euclDistance(vector1, vector2):  
    return sqrt(sum(power(vector2 - vector1, 2)))  
  
# init centroids with random samples  
def initCentroids(dataSet, k):  
    numSamples, dim = dataSet.shape  
    centroids = zeros((k, dim))  
    for i in range(k):  
        index = int(random.uniform(0, numSamples))  
        centroids[i, :] = dataSet[index, :]  
    return centroids  
  
# k-means cluster  
def kmeans(dataSet, k):  
    numSamples = dataSet.shape[0]  
    # first column stores which cluster this sample belongs to,  
    # second column stores the error between this sample and its centroid  
    clusterAssment = mat(zeros((numSamples, 2)))  
    clusterChanged = True  
  
    ## step 1: init centroids  
    centroids = initCentroids(dataSet, k)  
  
    while clusterChanged:  
        clusterChanged = False  
        ## for each sample  
        for i in xrange(numSamples):  
            minDist  = 100000.0  
            minIndex = 0  
            ## for each centroid  
            ## step 2: find the centroid who is closest  
            for j in range(k):  
                distance = euclDistance(centroids[j, :], dataSet[i, :])  
                if distance < minDist:  
                    minDist  = distance  
                    minIndex = j  
              
            ## step 3: update its cluster  
            if clusterAssment[i, 0] != minIndex:  
                clusterChanged = True  
                clusterAssment[i, :] = minIndex, minDist**2  
  
        ## step 4: update centroids  
        for j in range(k):  
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]  
            centroids[j, :] = mean(pointsInCluster, axis = 0)  
  
    print('Congratulations, cluster complete!')  
    return centroids, clusterAssment  
  
# show your cluster only available with 2-D data  
def showCluster(dataSet, k, centroids, clusterAssment):  
    numSamples, dim = dataSet.shape  
    if dim != 2:  
        print("Sorry! I can not draw because the dimension of your data is not 2!" ) 
        return 1  
  
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
    if k > len(mark):  
        print("Sorry! Your k is too large! please contact Zouxy") 
        return 1  
  
    # draw all samples  
    for i in xrange(numSamples):  
        markIndex = int(clusterAssment[i, 0])  
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])  
  
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  
    # draw the centroids 
    print(centroids)
    for i in range(k):  
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)  
  
    plt.show()  
    
    
def trainTestSplit(X,test_size=0.3):
    X_num=X.shape[0]
    train_index=list(range(X_num))
    test_index=[]
    test_num=int(X_num*test_size)
    for i in range(test_num):
        randomIndex=int(np.random.uniform(0,len(train_index)))
        test_index.append(train_index[randomIndex])
        del train_index[randomIndex]

    train=X[train_index] 
    test=X[test_index]
    return train,test  
    
    

def getHistgram(center,datas):
    #对数据进行编码
   
    row,col = center.shape    
    histgram = zeros(row,uint8)
    count=0
    histdata =[]
    for i in datas:
        data=array(i)
        repetdata = tile(data,(row,1))
        internal = (repetdata-center)
        internal2=internal*internal
        #得到所有的和
        #sumdata =map(sum,internal2)
        sumdata =list(map(sum,internal2))
        #求取得到最小的乘积
        #print sumdata
        index =sumdata.index(min(sumdata))
        #print "====================="
        #print index
        histdata.append(index)
        #print index
        histgram[index]+=1
    #import matplotlib.pyplot as plt
    #img=np.array(Image.open('d:/pic/lena.jpg').convert('L'))
    return histgram
    
    
def apa_coffecient(data,wave):
    mode = pywt.Modes.smooth
    datalen = int(pow(2,np.log2(len(data))))
    #print len(data)
    #print datalen
    w = pywt.Wavelet(wave)
    a = data[0:datalen]
    max_level=pywt.dwt_max_level(len(a),w)
    #print max_level
    coffefit=[]
    level = "level_%d"
    for i in range(max_level):
        (a,d) = pywt.dwt(a,w,mode)
        coffefit.extend(a)
    return coffefit
    
#generate code books
def GetLocalFeatures(segmentsize,step):
    #Extract local features
    fileIn = open('EPS.txt')
    dataSet=[]
    index=0
    for line in fileIn.readlines():  
        lineArr = line.strip().split(' ')  
        start=0
        #print index
        index+=1
        #print("Get local feature:%d" % index)
        flag = lineArr[0]
        lineArr= lineArr[2:]
        end = start+segmentsize
        while end<len(lineArr):
            data = lineArr[start:end]
            cofficient=apa_coffecient(data,'haar')
            dataSet.append(cofficient)
            start+=step
            end = start+segmentsize
    dataSet = mat(dataSet)
    #print dataSet.shape
    return dataSet

def GenerateCodeBooks(allfeature,seg,codebooksize):
    train,test=trainTestSplit(allfeature,test_size=0.05)
    clf = KMeans(n_clusters=codebooksize)
    s = clf.fit(test)
    #9个中心
    #print clf.cluster_centers_
    fileName = "center_%d_%d.csv" %(seg,codebooksize)
    savetxt(fileName,clf.cluster_centers_,delimiter=',')
    return clf.cluster_centers_
    
    
def representdata(inputdata,step,segmentsize,center):
    '''
    Encoding the data
    '''
    start=0
    end = start+segmentsize
    
    row,col = center.shape
    dataSet=[]
    while end<len(inputdata):
        data = inputdata[start:end]
        cofficient=apa_coffecient(data,'haar')
        #print cofficient
        dataSet.append(cofficient)
        start+=step
        end = start+segmentsize
    if len(dataSet)==0:
        return ''
    dataSet =mat(dataSet)
    histgram=getHistgram(center,dataSet)
    return histgram


