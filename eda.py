# -*- coding: utf-8 -*-
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import seaborn as sns

data = pd.read_csv('data/aug_train_cleaned.csv')

# Univariate Analysis
quantitative = {'cont':['city_development_index'],'discrete':['training_hours']}
qualitative = {'nominal':['city','gender','relevent_experience', 'enrolled_university', 'education_level',
                          'major_discipline', 'company_type'],
               'ordinal':['experience','last_new_job','min_company_size','max_company_size','avg_company_size']}

#Visualizing distribution for each quantitative variable
sns.displot(data[quantitative['cont'][0]],cumulative=False,kde=True,alpha=0.3,stat='probability')
#Quite obviously people with lesser city dev index left their job
sns.boxplot(x=data['target'],y=data[quantitative['cont'][0]])
sns.boxplot(x=data['target'],y=data[quantitative['cont'][0]],hue=data[qualitative['nominal'][2]])
sns.boxplot(x=data['target'],y=data[quantitative['cont'][0]],hue=data[qualitative['nominal'][1]])
sns.boxplot(x=data['target'],y=data[quantitative['cont'][0]],hue=data[qualitative['nominal'][3]])
sns.boxplot(x=data['target'],y=data[quantitative['cont'][0]],hue=data[qualitative['nominal'][4]])
sns.boxplot(x=data['target'],y=data[quantitative['cont'][0]],hue=data[qualitative['nominal'][5]])
sns.boxplot(x=data['target'],y=data[quantitative['cont'][0]],hue=data[qualitative['nominal'][6]])



sns.boxplot(x=data['target'],y=data[quantitative['discrete'][0]],hue=data[qualitative['nominal'][2]])





sns.histplot(data[quantitative['discrete'][0]],binwidth=(10),cumulative=True,element='poly',alpha=0.3,stat='probability')
sns.histplot(data[quantitative['discrete'][0]],binwidth=(10),cumulative=False,stat='count')

#Mine realtions b/w various quantitative features visually


def get_index(feature):
    return feature.value_counts().index
    
sns.countplot(data[qualitative['nominal'][1]])

sns.countplot(data[qualitative['nominal'][2]])

sns.countplot(data[qualitative['nominal'][3]])

sns.countplot(data[qualitative['nominal'][4]])

sns.countplot(y=data[qualitative['nominal'][5]])

sns.countplot(y=data[qualitative['nominal'][6]], order = get_index(data[qualitative['nominal'][6]]))

sns.countplot(data[qualitative['ordinal'][0]], order = get_index(data[qualitative['ordinal'][0]]))

sns.countplot(data[qualitative['ordinal'][1]])

sns.countplot(data[qualitative['ordinal'][-1]], order = get_index(data[qualitative['ordinal'][-1]]))

data[data['city']=='city_103']

for cols in data.columns:
    print(cols,"::\n",data[cols].value_counts())   




