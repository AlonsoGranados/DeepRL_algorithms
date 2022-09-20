import numpy as np

def create_tiling(feature_range, number_tilings, bins):
    table = np.zeros((number_tilings,bins+1))
    for i in range(number_tilings):
        table[i] = np.linspace(feature_range[0],feature_range[1],bins+1) \
                   + (i-1) * (feature_range[1]-feature_range[0])/number_tilings
    return table

def transform_data(feature,table):
    new_data = np.zeros((table.shape[0],table.shape[1]-1))
    for i in range(table.shape[0]):
        for j in range(table.shape[1]-1):
            if(feature < table[i,j]):
                break
            elif(table[i,j] <= feature and feature < table[i,j+1]):
                new_data[i,j] = 1
    return new_data

t = create_tiling([5,6],3,11)

print(transform_data(5.5,t))