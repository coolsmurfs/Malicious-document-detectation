import bagofwords
import staticinfo
import  WaveTransform as pyt 
import pandas as pd
import math
import numpy as np
import os
#读取wenjian

for (root,dirs,files) in os.walk(os.path.join(os.getcwd(),'new_centers')):
    for filet in files:
        segment = int(filet.split('_')[1])
        codebooksize=int(filet.split('_')[2][:-4])
        codefile = os.path.join(root,filet)
        codebook =np.loadtxt(open(codefile,"rb"),delimiter=",",skiprows=0)  
        #print codebook
        #codebook=bagofwords.GenerateCodeBooks(features,segment,codebooksize)
        #print(codebook)
        FileIndex=0
        
        #continue
        #编码数据
        #读取文件一行一行进行处理
        Features=[]
        fin = open("EPS.txt")
        for readdata in fin.readlines():
            
                sampleFeatures=[]
                data = readdata.split(' ')
                flag = data[0]
                inputdata = data[2:]
                inputdata=[float(dt) for dt in inputdata]
                stainfo = staticinfo.Stats(inputdata)
                #首先是统计信息
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
                #开始计算词袋模型特征
                FileIndex+=1
                print("segments:%d codebooksize:%d:"%(segment,codebooksize),FileIndex)
                histgram=bagofwords.representdata(inputdata,2,segment,codebook)
                #print histgram
                if len(histgram)<=0:
                    continue
                #print(histgram)
                for i in range(len(histgram)):
                    sampleFeatures.append(histgram[i])
                sampleFeatures.append(flag)
                Features.append(sampleFeatures) 
            
        columns = ['EntropyLength','sqLength','mean','stdev','maxvalue','maxpecent','zeropecent']
        
        for i in range(20):
            columns.append('Spream_%d' % i)
        for i in range(len(histgram)):
            columns.append('histgram_%d' % i)
        columns.append('flag')
        df2 =pd.DataFrame(Features,columns=columns)
        df2.to_csv("trainsmple%d_%d.csv" % (segment,codebooksize),index=False)

