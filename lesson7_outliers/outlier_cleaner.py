#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    for i in range(0, len(predictions)):
        cleaned_data.append((ages[i], net_worths[i], predictions[i] - net_worths[i]))
    
    cleaned_data.sort(key = lambda tup: tup[2], reverse = True)
    
    lower_index = int((len(cleaned_data) * 0.1) - 1)

    return cleaned_data[lower_index:]

