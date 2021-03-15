# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:56:55 2021

@author: Cosmos
"""
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer, StandardScaler
from sklearn.compose import ColumnTransformer

PATH = "data/aug_train_cleaned.csv"
data = pd.read_csv(PATH)

##Feature engineering. 

#Note for imbalanced data.

## Look for outliers and ways to handle them
def get_stds(feature):
    mean = feature.mean()
    std = feature.std()
    stds = {}
    for i in range(-3,4):
        if i != 0:
            stds[i] = round(mean + std * i,3)
    return stds

def get_bounds(feature,far_out_fact = 1.5):
    lower_quantile =  feature.quantile(0.25)
    upper_quantile =  feature.quantile(0.75)
    IQR = upper_quantile - lower_quantile
    lower_bounds = lower_quantile - (IQR * far_out_fact)
    upper_bounds = upper_quantile + (IQR * far_out_fact)
    return round(upper_bounds,3),round(lower_bounds,3)
    
upper,lower = get_bounds(data.city_development_index,1.5)

def count_outs():
    count = 0   
    for data_points in data.city_development_index:
        if data_points > upper or data_points < lower:  
            count+=1
    return count

upper,lower = get_bounds(data.city_development_index,1.5)

#consider only the lower bound as there are no ouliers beyon the upper bound.
data.loc[(data['city_development_index'] >= lower) & (data['city_development_index'] <= upper)]

## Encode features
nominal_features = ['gender','enrolled_university','major_discipline','company_type','education_level']
#ordinal_features = ['education_level']

# Lets add company size as categotical and evaluate performance later with cv
order = dict(zip(data['avg_company_size'].value_counts().sort_index().index.tolist(),[i for i in range(1,9)]))
data['avg_comp_size_cat'] = data['avg_company_size'].apply(lambda x: order[x])

#Pipelines for transforms
def binarize(data,feature):
    cats = data[feature].unique()
    return data[feature].apply(lambda x: 1 if x == cats[0] else 0)

data['relevent_experience']=binarize(data,'relevent_experience')     

def get_ohe_cols(data, nominal_features):
    col_list = list(data.columns)
    enc_list = []
    
    for index,col in enumerate(list(data.columns)):
        if col in nominal_features:
            attr_name = col
            col_list.remove(col)
            for app_idx in range(len(data[col].unique())):
               # col_list.insert(index+app_idx,attr_name+str(app_idx))
               #print(attr_name,"done")
               enc_list.append(attr_name+str(app_idx))
    return enc_list+col_list 

def get_tsfm_index(data,feature_list):
    return [idx for idx,i in enumerate(data.columns) if i in feature_list]


cols = get_ohe_cols(data, nominal_features)      
ohe_pipeline = ColumnTransformer([('encoder', OneHotEncoder(), get_tsfm_index(data, nominal_features)),
                                  ('scaler', StandardScaler(), get_tsfm_index(data, ['city_development_index']))], remainder='passthrough')
#le_pipeline = ColumnTransformer([('le_encoder', LabelEncoder(), [data.columns.get_loc("relevent_experience")])], remainder='passthrough')
ohed = pd.DataFrame(ohe_pipeline.fit_transform(data),columns=cols)

def get_dummies(data,ohe_variables=None):
    drop_vars = []
    for instance in ohe_variables:
        drop_vars.append(instance+str(len(data[instance].unique())-1))
    return drop_vars

final_data = ohed.drop(columns = get_dummies(data, nominal_features),axis=1)

#wo_company_type.to_csv("data/aug_train_cleaned.csv",index=False)


#encoded_data = le_pipeline.fit_transform(ohed)