#!/usr/bin/env python
# coding: utf-8

# In[9]:


from flask import Flask, request, jsonify
from flask_restful import reqparse, abort, Api, Resource

import numpy as np
import pandas as pd

import warnings
# warnings.filterwarnings("ignore")
import joblib
# import h5
from itertools import permutations
# import tensorflow as tf
# import keras
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
import json

import os
import logging
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
import requests

# app.url_map.strict_slashes = False
# api = Api(app)

repairdata = pd.read_excel('repairData.xlsx')


app = Flask(__name__)
# In[10]:

print('API is running')

def Freq_Model(data, DTC):

    dtcs = DTC.split(',')
    x = data['FAULTS'].str.contains(dtcs[0])
    for i in range (1, len(dtcs)):
        x = x | data['FAULTS'].str.contains(dtcs[i])

    df = data[x].copy()

    df.loc[:, 'RC_Count'] = df.groupby('ROOT CAUSE COMPONENT')['ROOT CAUSE COMPONENT'].transform('count')
    df.loc[:, 'FAILURE RATE'] = df['RC_Count']*100/len(df.index)
    
    RCs = df['ROOT CAUSE COMPONENT'].unique()
    Freq_data = pd.DataFrame()
    for RC in RCs:
        df_RC = df[df['ROOT CAUSE COMPONENT'] == RC].copy()

        PCs = df_RC['POSSIBLE CAUSE'].unique()
        df_RC.loc[:, 'PC_Count'] = df_RC.groupby('POSSIBLE CAUSE')['POSSIBLE CAUSE'].transform('count')
        df_RC.loc[:, 'PC WEIGHTAGE'] = df_RC['PC_Count']*100/len(df_RC.index)

        df_RC = df_RC.drop(['FAULTS'], axis = 1).drop_duplicates()
        Freq_data = Freq_data.append(df_RC)

    Freq_data = Freq_data.reindex(columns = ['ROOT CAUSE COMPONENT', 'FAILURE RATE', 'POSSIBLE CAUSE', 'PC WEIGHTAGE'])
    Freq_data = Freq_data.reset_index(drop = True)
    
    return Freq_data

@app.route('/feedbackloop', methods=["GET","POST"])
def predict():
    args = request.get_json()
    dtcs = args['dtcs']
    dtcs = re.sub("\(.*?\)", "", str(dtcs))
    dtc = re.sub("\s+", "", str(dtcs))

    Freq_data = Freq_Model(repairdata, dtc)
    flattened_dict1 = []
    
    df_json2 = json.loads(Freq_data.to_json())
    flattened_dict2 = []
    for i in range(len(df_json2["ROOT CAUSE COMPONENT"])):
        flattened_dict2.extend([{'rootCauseComponent': df_json2['ROOT CAUSE COMPONENT'][str(i)], 
                                'failureRate': df_json2['FAILURE RATE'][str(i)],
                                'possibleCause': df_json2['POSSIBLE CAUSE'][str(i)], 
                                'possibleCauseWeightage': df_json2['PC WEIGHTAGE'][str(i)]}])

    Output = {'partProbabilities':flattened_dict1, 'failureRates':flattened_dict2}

    return Output


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000)
