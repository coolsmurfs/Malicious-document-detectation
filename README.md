# Malicious-document-detectation
Detecting malicious document based on file's signal processing. Guide for using the code and dataset Generate the EPS.txt file The first step is to generate the EPS.txt file which contain the entropy sequence of the samples. The procedure can be as follows:

1.Open the file "GenerateETS.py" using python.
2.In line 6 of the code, change the folder where the path of the samples. For example, we can change “DataSet” to “D:\DataSet” if the dataset store in “D:\DataSet”. The row dataset can be download here “”
3.Run this script, then it will generate the ESP.txt. Here, we have generated an EPS.txt file which contain the entropy sequence of the samples. To save time, people can use the files that have been generated. Extract feature from the entropy sequence We can extract features and generate the training file when we generate ESP.txt file. 1、	Open the file “ExtractFeatures.py” using python. 2、	In the line 26 of the code, we must specific the name of the file which contain the entropy sequence of the samples. Here the default file name is “ESP.txt”. 3、	The folder “TrainDataSet” has contain parts of the training data (local segment is 6 and the size of the codebook range from 80 to 300 ), users can use them to training the module. The full training samples can be download http://.
4.Run the script, then it will generate the codebooks and then features file. The features file contains three categories of features: global features, DWT features and BOW features. The feature file will be named as “trainsmple X1_X2.csv”. Where the X1 is the length of the local segment and X2 is the length of codebooks. Train the module
5.Open the file “mal-document-classifier.py” using python.
6.In the line 112 of the code, specific the file name of the training data. Here the default file name is “trainsmple6_250.csv”.
7.Run the script, then it will generate the results.
8.The result will be showed in the figures.
