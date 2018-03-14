# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 07:51:27 2018

@author: BarraultMJ

This approach uses the AutoML TPOT to investigate numerous pipelines 

http://epistasislab.github.io/tpot/using/
"""

import sys
import pickle
import pandas as pd
import numpy as np
import time

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import load_classifier_and_data, test_classifier

from tpot import TPOTClassifier
from sklearn.cross_validation import train_test_split

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

features_list = list(next(iter(data_dict.values())).keys())
features_list.remove('poi')
features_list.insert(0, 'poi')

features_list.remove('email_address')

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, remove_NaN = True, sort_keys = True)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# TPOT expects the input features to be np.array
# moved this type conversion to targetFeatureSplit()
#features_train = np.array(features_train)
#features_test = np.array(features_test)
#labels_test = np.array(labels_test)
#labels_train = np.array(labels_train)

# , warm_start = True
# changes scoring metric to f1 since this a binary classification task and
# we are looking to maximize both precision and recall. Using accuracy only
# was producing pipelines that were overfitted to the training data
tpot = TPOTClassifier(scoring='f1', verbosity=2)
start = time.clock()
tpot.fit(features_train, labels_train)
end = time.clock()
print('Optimisation took {:.2f} seconds'.format(end - start))
print('Optimised pipeline score: {:.4f}'.format(tpot.score(features_test, labels_test)))

tpot.export('poi_best_pipeline.py')

# Test exported pipeline

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Binarizer
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator

## NOTE: Make sure that the class is labeled 'target' in the data file
#tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
#features = tpot_data.drop('target', axis=1).values
#training_features, testing_features, training_target, testing_target = \
#            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:0.6757575757575757
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.05, min_samples_leaf=1, min_samples_split=14, n_estimators=100)),
    Binarizer(threshold=0.1),
    RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.25, n_estimators=100), step=0.35000000000000003),
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="entropy", max_depth=9, min_samples_leaf=2, min_samples_split=3)),
    StackingEstimator(estimator=GaussianNB()),
    RBFSampler(gamma=0.4),
    BernoulliNB(alpha=0.001, fit_prior=True)
)

# End of exported pipeline

#exported_pipeline.fit(features_train, labels_train)
#results = exported_pipeline.predict(features_test)
#score = exported_pipeline.score(features_test, labels_test)
test_classifier(exported_pipeline, my_dataset, features_list)