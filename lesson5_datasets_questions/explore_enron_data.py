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

# What’s the value of stock options exercised by Jeffrey Skilling?
enron_data["SKILLING JEFFREY K"]['exercised_stock_options']

# Of these three individuals (Lay, Skilling and Fastow), who took home the most 
# money (largest value of “total_payments” feature)? 
l = ["SKILLING JEFFREY K", 'FASTOW ANDREW S', 'LAY KENNETH L']

for poi in l:
    print poi + " " + str(enron_data[poi]['total_payments'])
    
# How many folks in this dataset have a quantified salary?
# What about a known email address?

num_sal = 0
num_email = 0

for key, value in enron_data.iteritems():
    #print value
    if not (value['salary'] == 'NaN'):
        num_sal += 1
    if not (value['email_address'] == 'NaN'):
        num_email += 1

print "number of salaries " + str(num_sal)
print "number of email addresses " + str(num_email)

# How many people in the E+F dataset (as it currently exists) 
# have “NaN” for their total payments? What percentage of people in the 
# dataset as a whole is this?
total_people = len(enron_keys)
total_payments_count = [enron_data[i]["total_payments"] \
                        for i in enron_data.keys() \
                        if enron_data[i]["total_payments"] == 'NaN'] 
people_payments_count = len(total_payments_count)
percent_payments = (people_payments_count / float(total_people)) * 100.0
print percent_payments
    
# How many POIs in the E+F dataset have “NaN” for their total payments? 
# What percentage of POI’s as a whole is this?

total_people = len(enron_keys)
poi_total_payments_count = [enron_data[i]["total_payments"] \
                        for i in enron_data.keys() \
                        if enron_data[i]["total_payments"] == 'NaN' and \
                        enron_data[i]["poi"] == True] 
poi_payments_count = len(poi_total_payments_count)
poi_percent_payments = (poi_payments_count / float(total_people)) * 100.0
print poi_percent_payments
    