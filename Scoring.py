import json
import numpy as np
import os
import pickle
import joblib
from azureml.core import Model

def init():

    global model
    model_name = 'model-iris'
    #path = Model.get_model_path(model_name, version=1)
    print(os.getenv('AZUREML_MODEL_DIR'))
    path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'iris.pkl')
    
    ## Another way to load the pickle file, I was getting some error, due to some path mismatch, use above syntax
    ##path = Model.get_model_path(model_name)
    
    print(path)
    model = joblib.load(path)

def run(data):

    try:
        data = json.loads(data)
        result = model.predict(data['data'])
        return {'data' : result.tolist() , 'message' : "Successfully classified Iris"}

    except Exception as e:
        error = str(e)
        return {'data' : error , 'message' : 'Failed to classify iris'}