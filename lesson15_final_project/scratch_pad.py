# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 08:20:30 2018

@author: BarraultMJ
"""

# Quick statistical exploration of poi data set
import pprint as pp
from scipy import stats

def compare_pois(pois, npois):
    results = {}
    for c in pois:
        results[c] = {"poi": pois[c].mean(), "npoi": npois[c].mean(), \
               "delta": pois[c].mean() - npois[c].mean(), "t": stats.ttest_ind(pois[c], npois[c])}
    return results

pois = []
for p in my_dataset:
    print(type(my_dataset[p]['bonus']))
    if my_dataset[p]['poi'] == True:
        temp_poi = my_dataset[p]
        temp_poi['name'] = p
        pois.append(temp_poi)

npois = []
for p in my_dataset:
    print(type(my_dataset[p]['bonus']))
    if my_dataset[p]['poi'] == False:
        temp_npoi = my_dataset[p]
        temp_npoi['name'] = p
        npois.append(temp_npoi)

pois_df = pd.DataFrame(pois)
npois_df = pd.DataFrame(npois)

pois_df = pois_df.apply(pd.to_numeric, errors='coerce')
npois_df = npois_df.apply(pd.to_numeric, errors='coerce')

pois_df = pois_df.fillna(0)
npois_df = npois_df.fillna(0)

pp.pprint(compare_pois(pois_df, npois_df))

# exercised_stock_options has the smallest pvalue

"""
Working with TPOT

features should be a 2D np array
still should use train / test split
mising xgboost package, installed it
ran several evolutions at small values for generations and populations
ran several evolutions at 
"""
