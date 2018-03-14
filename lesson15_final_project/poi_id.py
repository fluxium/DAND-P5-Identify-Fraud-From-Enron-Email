#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
from scipy import stats
import pprint as pp

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
# Python 3 refactor test
from tester import load_classifier_and_data, dump_classifier_and_data, test_classifier

"""
Compares the means of all available features in the Enron dataset between the
persons of interest group and the non persons of interest group
"""
def compare_pois(data):
    pois = []
    for p in data:
        if data[p]['poi'] == True:
            temp_poi = data[p]
            temp_poi['name'] = p
            pois.append(temp_poi)
    
    npois = []
    for p in data:
        if data[p]['poi'] == False:
            temp_npoi = data[p]
            temp_npoi['name'] = p
            npois.append(temp_npoi)
    
    pois_df = pd.DataFrame(pois)
    npois_df = pd.DataFrame(npois)
    
    pois_df = pois_df.apply(pd.to_numeric, errors='coerce')
    npois_df = npois_df.apply(pd.to_numeric, errors='coerce')
    
    pois_df = pois_df.fillna(0)
    npois_df = npois_df.fillna(0)
    
    results = {}
    for c in pois_df:
        results[c] = {"poi": pois_df[c].mean(), "npoi": npois_df[c].mean(), \
               "delta": pois_df[c].mean() - npois_df[c].mean(), \
               "t": stats.ttest_ind(pois_df[c], npois_df[c])}
    pp.pprint(results)
    return results

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
    
### Task 2 (reordered for my analysis): Remove outliers
data_dict.pop('TOTAL', 0)

### Task 3 (reordered for my analysis): Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# exercised_stock_options has the smallest pvalue
compare_pois(my_dataset)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'exercised_stock_options'] # You will need to use more features

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

# Python 3 refactor test
clf_import, dataset_import, features_import = load_classifier_and_data()
