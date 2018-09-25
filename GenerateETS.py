import calc_array
import os

Entropyclc = calc_array.calc_array()
index=0
dirpath = os.path.join(os.getcwd(),'DataSet')
outData = open("ESP1.txt",'w')
for root,subdir,files in os.walk(dirpath):
    for filet in files:
        filepath=''
        flag=0
        filepath = os.path.join(root,filet)
        if "malware" in filepath:
            flag=1
        elif "clean" in filepath:
            flag=0
        entropy = Entropyclc.generate(filepath)
        print(entropy)
        data = "%s %d" % (flag,len(entropy))
        for i in range(len(entropy)):
            data+=" %f"% entropy[i]
        data+='\n' 
        outData.write(data)
outData.close()
      
