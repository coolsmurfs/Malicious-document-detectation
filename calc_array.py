# -*- coding: cp936 -*-
import os
import os.path
import zipfile
import numpy
from math import *  
import shutil

#bsp=bow.BoWSp()
#import matplotlib.pyplot as plt


features = []

class calc_array:

    '''Generate Time series'''
    
    def handle_rarfile(self,filename):
        outstream=''
        flag=False
        
        tempdir = os.path.join(os.getcwd(),'tempdir')
        if os.path.exists(tempdir):
            
            shutil.rmtree(tempdir)
        if not os.path.exists(tempdir):
            os.mkdir(tempdir)
        dest = os.path.join(tempdir,"temp.rar")
        shutil.copy(filename,dest)
        outstream=self.getrarfilestream(dest,tempdir)
        #print outstream
            
        return outstream

    def handle_zipFile(self,filename):
        '''
        input:filename
        output:the file stream
        '''
        #print filename
        #print(filename)
        outstream=bytes([])
        flag=False
        #try:
        try:
            tempdir = os.path.join(os.getcwd(),'tempdir')
            if os.path.exists(tempdir):
                
                shutil.rmtree(tempdir)
                #print(bsdf)
            if not os.path.exists(tempdir):
                os.mkdir(tempdir)
            
            #dest = os.path.join(tempdir,"temp.rar")
            zp = zipfile.ZipFile(filename,'r')
            zp.extractall(tempdir)
            #print(sdfsdfsdf)
            outfiles = self.walkdir(tempdir)
            #print(outfiles)
            for i in outfiles:
                fin = open(i,'rb')
                    #print(i)
                outstream+=fin.read()
                fin.close()
        except:
            outstream=bytes([])
        return outstream
        

    def getrarfilestream(self,zfile,path):
        outstream = bytes([])
        rar_command1 ="unrar x %s" % (zfile)
        
        #print(rar_command1)
        parpath = os.getcwd()
        os.chdir(path)
		 #Windows system
        #rar_command1 =r'"C:\Program Files (x86)\WinRAR\WinRAR.exe" x -ibck %s %s'%(zfile,path)
        if os.system(rar_command1)==0:
            os.remove(zfile)
            outfiles = self.walkdir(path)
            for i in outfiles:
                fin = open(i,'rb')
                #print(i)
                data=fin.read()
                #print(type(data))
                outstream+=data
                fin.close()
        os.chdir(parpath)
        print(os.getcwd())
        #print(outstream)
        return outstream
            

    def walkdir(self,inputdir,extension=None):
        '''
        walk through the inputdir and get the dir file
        '''
        outfiles=[]
        for parent,dirnames,filenames in os.walk(inputdir):
            for filename in filenames:
                if extension and filename.endswith(extension):
                    outfiles.append(os.path.join(parent,filename))
                else:
                    outfiles.append(os.path.join(parent,filename))
        return outfiles
        
    def calc_ent(self,x):
        '''
    calculate shanno ent of x
    '''
        x_value_list = set(x[i] for i in range(x.shape[0]))
        #print x_value_list
        ent = 0.0
        for x_value in x_value_list:
            p = float(x[x== x_value].shape[0]) / x.shape[0]
            logp = numpy.log2(p)
            ent -=p*logp
        return ent

    def GetFileNameAndExt(self,filename):
        data=''
        with open(filename,'rb') as f:
            data = f.read()
            f.close()
        if len(data)<2:
            print("we return======")
            return 'None'
        if(data[0]==0x52 and data[1]==0x61 and data[2]==0x72):
            exetension='docx'
        elif(data[0]==0x50 and data[1]==0x4B and data[2]==0x03):
            exetension='docx'
        elif(data[0]==0x7B and data[1]==0x5C):
            exetension='rtf'
        elif(data[0]==0xD0 and data[1]==0xCF and data[2]==0x11):
            exetension='ole'
        elif(data[0]==0x25 and data[1]==0x50 and data[2]==0x44):
            exetension='pdf'
        else:
            exetension='None'
        print(exetension)
        return exetension
    def generate(self,filename):
        '''
        if filename.endswith(".png"):
            print 'png'
            continue
        '''
        entropy_chunks=[]
        exetension= self.GetFileNameAndExt(filename)
        if exetension=='None':
            print(r"this system can only be use 'doc','rtf','ole', file ")
            print("Return ====== file zip")
            return None
        fp = open(filename,'rb')
        data=fp.read()
        fp.close()
        filedata=''
        if (data[0]=="\x52" and data[1]=="\x61" and data[2]=="\x72"):
            filedata=self.handle_rarfile(filename)
        if (data[0]=="\x50" and data[1]=="\x4B" and data[2]=="\x03"):
            filedata=self.handle_zipFile(filename)
        else:
            filedata=data
        if len(filedata)==0:
            return None
        remain_data = len(filedata)%256
        chunks = int(len(filedata)/256)
        finalData=''
        if remain_data<128:
            finalData = filedata[0:chunks*256]
        else:
            finalData = filedata+b'\x00'*(256-remain_data)
            chunks=chunks+1
        for i in range(chunks):
            data = [k for k in finalData[i*256:(i+1)*256]]
            data = numpy.array(data)
            entropy_chunks.append(self.calc_ent(data))
        #print("==========================Here return===")
        #print(len(entropy_chunks))
        
        return entropy_chunks
