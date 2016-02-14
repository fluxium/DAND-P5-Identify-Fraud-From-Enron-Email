#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../lesson15_final_project/final_project_dataset.pkl", "r"))

enron_data[enron_data.keys()[0]].keys()

enron_keys = enron_data.keys()
enron_keys.sort()

# How many people are there in the dataset
len(enron_data)

# How many features per person
len(enron_data["SKILLING JEFFREY K"])

# How many POIs
poi = [enron_data[ii]["poi"] for ii in enron_data.keys() if enron_data[ii]["poi"] == 1]
len(poi)

# How many POI total
f = open('../lesson15_final_project/poi_names.txt', 'r')
x = f.readlines()
print len(x) - 2 # Account for the two lines at the start of the file

# What is the total value of the stock belonging to James Prentice?
enron_data["PRENTICE JAMES"]['total_stock_value']

# How many email messages do we have from Wesley Colwell to persons of interest?
enron_data["COLWELL WESLEY"]['from_this_person_to_poi']