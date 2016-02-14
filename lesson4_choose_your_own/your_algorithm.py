#!/usr/bin/python
import sys
sys.path.append("../tools/")
from time import time
import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

# Try AdaBoost

def adaboost(n_estimators, learning_rate):
    
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.metrics import accuracy_score
    
    clf = AdaBoostClassifier(n_estimators = n_estimators, 
                             learning_rate = learning_rate)
    clf.fit(features_train, labels_train)
    
    t_fit = time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time()-t_fit, 3), "s"
    
    t_pred = time()
    pred = clf.predict(features_test)
    print "predict time:", round(time()-t_pred, 3), "s"
    
    print accuracy_score(pred, labels_test)

    try:
        prettyPicture(clf, features_test, labels_test)
    except NameError:
        pass

# Try Random Forest

def ranforest(n_estimators, min_samples_split):
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    clf = RandomForestClassifier(n_estimators = n_estimators, 
                                 min_samples_split = min_samples_split,
                                 bootstrap = True)
    clf.fit(features_train, labels_train)
    
    t_fit = time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time()-t_fit, 3), "s"
    
    t_pred = time()
    pred = clf.predict(features_test)
    print "predict time:", round(time()-t_pred, 3), "s"
    
    print accuracy_score(pred, labels_test)

    
    try:
        prettyPicture(clf, features_test, labels_test)
    except NameError:
        pass

def supportvector(C, gamma = 'default'):
    
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    
    if gamma == 'default':
        clf = SVC(kernel="rbf", C = C)
    else:
        clf = SVC(kernel="rbf", C = C, gamma = gamma)
    
    clf.fit(features_train, labels_train)
    
    t_fit = time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time()-t_fit, 3), "s"
    
    t_pred = time()
    pred = clf.predict(features_test)
    print "predict time:", round(time()-t_pred, 3), "s"
    
    print accuracy_score(pred, labels_test)
    
    try:
        prettyPicture(clf, features_test, labels_test)
    except NameError:
        pass
    