# -*- coding: utf-8 -*-
import os
import os.path
import numpy as np
#import matplotlib.pyplot as plt 
#import pylab as pl
import pywt
import pywt.data            
#计算小波能量谱   
mode = pywt.Modes.smooth
def calc_energyspectrum(data,wave):
    '''
data:要进行小波分解的能量数据
w:小波名称
    '''
    #When we calc the cofficient of the wavelet transfpom it requeie the length must be 2^x，
    #so we calc the datelen is 2*log2(T),where T is the length of data
  
    datalen = int(pow(2,np.log2(len(data))))
    #print len(data)
    #print datalen
    w = pywt.Wavelet(wave)
    a = data[0:datalen]
    #print a
    #energy_spectrum=[]
    #$ca= [] #
    #cd = []#detail cofficient the de wavelet decompress
    max_level=pywt.dwt_max_level(len(a),w)
    #print max_level
    coeffs=pywt.wavedec(data,'haar',level=max_level)
    #print len(coeffs)
    coeffs = list(reversed(coeffs))[:-1]
    #print "{}}}}}}}}}}}}}}"
    #print coeffs
    energy_specum = [0]*16
    for i in range(max_level):
        energy_specum[i]=sum([dcf*dcf for dcf in coeffs[i]])
    #print("=-------------------=====")
    #print(energy_specum)
    return energy_specum,max_level      
    #spectrum_a={}
    #spectrum_d={}
    #spectrum={}
    #cofficients = {}
    #level = "level_%d"
    #for i in range(8):
    #    (a,d) = pywt.dwt(a,w,mode)
    #    ca.append(a)
    #    cd.append(d)
    #    cofficients[level % i]=(a,d)
        #cofficients.append((a,d))
    #计算能量谱，密度
    '''
    outdict['d_level_0'] = sum([i*i for i in cofficients['level_0'][0]])
    outdict['d_level_1'] = sum([i*i for i in cofficients['level_1'][0]])
    outdict['d_level_2'] = sum([i*i for i in cofficients['level_2'][0]])
    outdict['d_level_3'] = sum([i*i for i in cofficients['level_3'][0]])
    outdict['d_level_4'] = sum([i*i for i in cofficients['level_4'][0]])
    outdict['d_level_5'] = sum([i*i for i in cofficients['level_5'][0]])
    outdict['d_level_6'] = sum([i*i for i in cofficients['level_6'][0]])
    outdict['d_level_7'] = sum([i*i for i in cofficients['level_7'][0]])
    outdict['d_level_8'] = sum([i*i for i in cofficients['level_0'][1]])
    outdict['d_level_9'] = sum([i*i for i in cofficients['level_1'][1]])
    outdict['d_level_10'] = sum([i*i for i in cofficients['level_2'][1]])
    outdict['d_level_11'] = sum([i*i for i in cofficients['level_3'][1]])
    outdict['d_level_12'] = sum([i*i for i in cofficients['level_4'][1]])
    outdict['d_level_13'] = sum([i*i for i in cofficients['level_5'][1]])
    outdict['d_level_14'] = sum([i*i for i in cofficients['level_6'][1]])
    outdict['d_level_15'] = sum([i*i for i in cofficients['level_7'][1]])
    #spectrum_a['level_0_d'] = sum([i*i for i in cofficients['level_0'][1]])
    #spectrum_a['level_0_d'] = sum([i*i for i in cofficients['level_0'][1]])
    '''
    #return outdict
    
def calc_energyspectrumrow(data,wave):
    '''
data:要进行小波分解的能量数据
w:小波名称
    '''
    #When we calc the cofficient of the wavelet transfpom it requeie the length must be 2^x，
    #so we calc the datelen is 2*log2(T),where T is the length of data
  
    datalen = int(pow(2,np.log2(len(data))))
    #print len(data)
    #print datalen
    w = pywt.Wavelet(wave)
    a = data[0:datalen]
    outdict=[]
    #print a
    #w = pywt.Wavelet(wave)
    max_level=pywt.dwt_max_level(len(a),w)
    
    #energy_spectrum=[]
    ca= [] #
    cd = []#detail cofficient the de wavelet decompress
    #cofficients = {}
    #level = "level_%d"
    for i in range(max_level):
        (a,d) = pywt.dwt(a,w,mode)
        ca.append(a)
        cd.append(d)
        #cofficients[level % i]=(a,d)
    outdict1=[0]*20
    outdict2=[0]*20
    #print(cd)
    index=0
    for coa in range(len(ca)):
        if index>19:
            break
        outdict1[index]=(sum([k*k for k in ca[coa]]))
        index+=1
    indexb=0
    for cod in range(len(cd)):
        if indexb>19:
            break
        outdict2[indexb]=(sum([k*k for k in cd[cod]]))
        indexb+=1
    #print("+++++++++++++++++++++++++++++++++++++++============================")
    #print(outdict2)
    return outdict1,outdict2
        #outdict.append(sum([i*i for i in cofficients['level_0'][0]]))
        #cofficients.append((a,d))
    #计算能量谱，密度
    '''
    outdict['a_level_0'] = sum([i*i for i in cofficients['level_0'][0]])
    outdict['a_level_1'] = sum([i*i for i in cofficients['level_1'][0]])
    outdict['a_level_2'] = sum([i*i for i in cofficients['level_2'][0]])
    outdict['a_level_3'] = sum([i*i for i in cofficients['level_3'][0]])
    outdict['a_level_4'] = sum([i*i for i in cofficients['level_4'][0]])
    outdict['a_level_5'] = sum([i*i for i in cofficients['level_5'][0]])
    outdict['a_level_6'] = sum([i*i for i in cofficients['level_6'][0]])
    outdict['a_level_7'] = sum([i*i for i in cofficients['level_7'][0]])
    outdict['d_level_0'] = sum([i*i for i in cofficients['level_0'][1]])
    outdict['d_level_1'] = sum([i*i for i in cofficients['level_1'][1]])
    outdict['d_level_2'] = sum([i*i for i in cofficients['level_2'][1]])
    outdict['d_level_3'] = sum([i*i for i in cofficients['level_3'][1]])
    outdict['d_level_4'] = sum([i*i for i in cofficients['level_4'][1]])
    outdict['d_level_5'] = sum([i*i for i in cofficients['level_5'][1]])
    outdict['d_level_6'] = sum([i*i for i in cofficients['level_6'][1]])
    outdict['d_level_7'] = sum([i*i for i in cofficients['level_7'][1]])
    #spectrum_a['level_0_d'] = sum([i*i for i in cofficients['level_0'][1]])
    #spectrum_a['level_0_d'] = sum([i*i for i in cofficients['level_0'][1]])
    '''
    #return outdict,max_level
            
