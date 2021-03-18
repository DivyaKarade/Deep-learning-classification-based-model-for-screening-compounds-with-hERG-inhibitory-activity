# Deep-learning-classification-based-model-for-screening-compounds-with-hERG-inhibitory-activity

Developing a Deep learning classification-based model for screening pharmaceutical compounds with hERG inhibitory activity (cardiotoxicity) and using the model to screen CAS antiviral database to identify compounds with cardiotoxicity potential.  

The data is taken from "Drug Discovery Hackathon 2020: PS ID: DDT2-13" (https://innovateindia.mygov.in/ddh2020/problem-statements/)  

Details related to the project can also be derived from: (https://youtu.be/7tqaPmYQmCM)  

Note: The solution for the above problem statement is solved with Deep learning classification based model instead of linear discriminant analysis model as written in the problem statement.  

Details of the project: 
In silico prediction of cardiotoxicity with high sensitivity and specificity for potential drug molecules would be of immense value. Hence, building a classification-based machine learning models, capable of efficiently predicting cardiotoxicity will be critical. A data set of diverse pharmaceutical compounds with hERG channel inhibitory activity (blocker/non-blocker) is provided.  The SMILES notations of all compounds are given. The set of compounds divided into a training set and a test set using 70:30 ratios. Simple, reproducible and easily transferable classification models developed from the training set compounds using 2D descriptors. The models were validated based on the test set compounds.  
The models is having the following quality:  
Training Set: 
Classification accuracy for training set: 0.986058
Precision for training set: 0.993124 
Sensitivity/Recall for training set: 0.990235 
F1 score for training set: 0.991677 
ROC AUC for training set: 0.977280 
Confusion matrix: 
[[ 892   33]  
[  47 4766]]  

Test set: 
 Classification accuracy for test set: 0.813670 
Precision for test set: 0.883061 
Sensitivity/Recall for test set: 0.990235 
F1 score for test set: 0.889050 
ROC AUC for test set: 0.649767
Confusion matrix: 
[[ 165  243]  
[ 215 1835]]  

The best model was also used to classify CAS antiviral database compounds for hERG channel inhibitory activity and a list of compounds with cardiotoxicity potential was being generated in the form of .csv file.
