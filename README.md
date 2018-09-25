Dear Researcher,

Thank you for using this code and datasets. I explain how the code related to my paper " Capturing the symptoms of malicious code in electronic documents by file’s entropy signal combined with Machine learning " published in Applied Soft Computing, works. The main datasets mentioned in the paper together with code are included.
If there is any question, feel free to contact me at:
lupingllp@gmail.com

Regards,

Liuping liu


Noting: because github limits the size of uploading files, then we have store the full data and code at google's drive. The address is here:https://drive.google.com/drive/folders/1qs0pTcrvfCNcmPXq6GkaDxMdf2FwB0s8?usp=sharing.

Preparation:
The code is implemented with python script and some of packages in python are used. Therefore, we must configure the environment before we run the code.
1、	The code runs in python 2.7. Then we must install the python interpreter first.
2、	We can use the command “pip install package name” to install the dependency. The command used to install packages are as follows:
	pip install numpy （ubuntu: sudo pip install numpy）
	pip install pandas (ubuntu:sudo pip install pandas)
	pip install matplotlib (ubuntu: sudo apt-get install python-matplotlib)
	pip install sklearn (Ubuntu:sudo apt-get install sklearn)
	pip install scipy (Ubuntu: sudo pip install scipy)
	pip install xgboost (Ubuntu: sudo pip install xgboost)
	pip install PyWavelets (Ubuntu:sudo pip install PyWavelets)
Users of the Anaconda python distribution may wish to obtain those packages by using the command “conda install package name”.

Guidelines for Code and DataSet:

Generate the EPS.txt file
      The first step is to generate the EPS.txt file which contain the entropy sequence of the samples. The procedure can be as follows:
1.	Open the file "GenerateETS.py" using python.
2.	In line 6 of the code, change the folder where the path of the samples. For example, we can change “DataSet” to “D:\DataSet” if the dataset store in “D:\DataSet”. The row dataset can be download here https://github.com/coolsmurfs/Malicious-document-detectation.
3.	Run this script, then it will generate the ESP.txt. Here, we have generated an EPS.txt file which contain the entropy sequence of the samples. To save time, people can use the files that have been generated.
Extract feature from the entropy sequence
	We can extract features and generate the training file when we generate ESP.txt file.
1、	Open the file “ExtractFeatures.py” using python.
2、	In the line 26 of the code, we must specific the name of the file which contain the entropy sequence of the samples. Here the default file name is “ESP.txt”.
3、	The folder “TrainDataSet” has contain parts of the training data (local segment is 6 and the size of the codebook range from 80 to 300 ), users can use them to training the module. The full training samples can be download https://github.com/coolsmurfs/Malicious-document-detectation.
4.	Run the script, then it will generate the codebooks and then features file. The features file contains three categories of features: global features, DWT features and BOW features. The feature file will be named as “trainsmple X1_X2.csv”. Where the X1 is the length of the local segment and X2 is the length of codebooks.
Train the module
1.	Open the file “mal-document-classifier-training.py” using python.
2.	In the line 112 of the code, specific the file name of the training data. Here the default file name is “trainsmple6_250.csv”.
3.	Run the script, then it will generate the results.
4.	The result will be showed in the figures. 
5.	t
Test the module
1.	When we have trained the module, we can use it to predict new documents.
2.	Here we have trained a model named "RFClassifier.m". Users can use this model to predict new files.
3.	Open a command line windows in the target directory, then run the command "python TestingModule.py", it will ask the user to input the target file's path. When users input the target's path, it will give the result.
4.	The dataset in the folder TestData is the samples used to compare with the anti-virus engines and prevalent tools.
 
Figure 1 detecting malicious documents

 
Figure 2 detecting clean documents



