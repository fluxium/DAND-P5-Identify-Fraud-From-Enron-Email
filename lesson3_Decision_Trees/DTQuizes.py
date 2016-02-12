#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
sys.path.append("../tools/")
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from classifyDT import classify

features_train, labels_train, features_test, labels_test = makeTerrainData()

from sklearn import tree
from sklearn.metrics import accuracy_score

clf_split_2 = tree.DecisionTreeClassifier(min_samples_split = 2)
clf_split_50 = tree.DecisionTreeClassifier(min_samples_split = 50)

clf_split_2.fit(features_train, labels_train)
clf_split_50.fit(features_train, labels_train)

pred_split_2 = clf_split_2.predict(features_test)
pred_split_50 = clf_split_50.predict(features_test)

acc_min_samples_split_2 = accuracy_score(pred_split_2, labels_test)
acc_min_samples_split_50 = accuracy_score(pred_split_50, labels_test)
### be sure to compute the accuracy on the test set

def submitAccuracies():
  return {"acc":round(acc,3)}

#### grader code, do not modify below this line

prettyPicture(clf_split_2, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
prettyPicture(clf_split_50, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
