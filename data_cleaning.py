# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import seaborn as sns
PATH_TRAIN = "data/aug_train.csv"
PATH_TEST = "data/aug_test.csv"

df = pd.read_csv(PATH_TRAIN)

#Enrollee is unique
print(len(df.enrollee_id.unique()))

#Imabalnced city distribution
print(df.city.value_counts())

#--Tasks--
#Clean Company Size 
def clean_company_size(x,agg):
    x = str(x)
    minim,maxim=0,0
    try:
        if '+' in x:
            minim = x[:-1]
            maxim = 20000 #random.randrange(10010,20000,1000)
        elif '<' in x:
            minim = 1.0
            maxim = x[1:]
        elif '-' in x:
            minim = x.split('-')[0]
            maxim = x.split('-')[1]
        elif '/' in x:
            minim = x.split('/')[0]
            maxim = x.split('/')[1]
    except:
        print("Failed to handle external cases!")
    if maxim==0:
        return np.nan
    if agg=='min':
        return float(minim)
    elif agg=='max':
        return float(maxim)
    elif agg=='avg':
        return float((float(maxim)+float(minim))/2)
    else:
        print("invalid aggregate")
df['min_company_size'] = df['company_size'].apply(lambda x:clean_company_size(x,'min'))
df['max_company_size'] = df['company_size'].apply(lambda x:clean_company_size(x,'max'))
df['avg_company_size'] = df['company_size'].apply(lambda x:clean_company_size(x,'avg'))

#Remove obs with minimal nansc
ignore_nans = ['enrolled_university','education_level','experience','last_new_job']
def del_min_nans(ignore_nans):
    filtered = df
    for cols in ignore_nans:
        filtered = filtered[filtered[cols].isnull()==False]
        print(filtered)
    return filtered
d = del_min_nans(ignore_nans)

## Gender Imputation
# Add new category to the gender column 
d['gender']=d['gender'].apply(lambda x: "Non disclosure" if x is np.nan else x)

# Major discipline imputation with most freequent varialbe(distribution highly 
# inclined towards STEM).
d['major_discipline'].fillna(d.major_discipline.mode()[0],inplace=True)

# Company Type imputation
# Attempt to impute by knn with city dev index
    # Company type Encode for knn
# =============================================================================
#     temp_comp_encode = pd.get_dummies(d.company_type,dummy_na=True)
#     temp_comp_encode['nans'] = temp_comp_encode.iloc[:,-1]
#     temp_comp_encode.drop(np.nan,axis=1,inplace=True)
#     temp_comp_encode.loc[temp_comp_encode.nans == 1, temp_comp_encode.columns] = np.nan
# =============================================================================
# =============================================================================
# 
# features = d.city_development_index
# knn_imputer = KNNImputer(n_neighbors=5)
# knn_imputer.fit_transform(np.c_[np.asarray(features),np.asarray(temp_comp_encode)])
# =============================================================================

# KNN-imputation also suggests that for almost all cases PVT Ltd is recommened for fillna
d['company_type'].fillna(d.company_type.mode()[0],inplace=True)

# Average company size makes more sense over here.
# Since variable is numerically discrete and the distribution is not dominated by a single
# category lets add a column for keeping track of nans.
wo_company_type = d.drop(columns=['company_size'],axis=1)
wo_company_type['is_avg_nan'] = wo_company_type['avg_company_size'].apply(lambda x: True if np.isnan(x) else False) 
wo_company_type['avg_company_size'].fillna(wo_company_type.avg_company_size.mode()[0],inplace=True)

wo_company_type.to_csv("data/aug_train_cleaned.csv")

