#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel="rbf", C = 10000.0)

# Improving the performance, reducing accuracy
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

t_fit = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t_fit, 3), "s"

t_pred = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t_pred, 3), "s"

print accuracy_score(pred, labels_test)

# clf.score() was getting confused about the shape of the inputs
# switched to using accuracy_score
#clf.score(pred, labels_test)
#########################################################


