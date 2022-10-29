import json
from collections import namedtuple
import io

import numpy as np
import pandas as pd
import torch

import sys
import logging
from torch import nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

###################################
### VARIABLES 
###################################

MODEL_NAME = 'model.pth'

###################################
### SAGEMAKER LOAD MODEL FUNCTION 
###################################   

# You need to put in config.json from saved fine-tuned Hugging Face model in code/ 
# Reference it in the inference container at /opt/ml/model/code
def model_fn(model_dir):
    model = torch.nn.Linear(5, 1)
    model_path = '{}/{}'.format(model_dir, MODEL_NAME) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model

###################################
### SAGEMAKER PREDICT FUNCTION 
###################################   

def predict_fn(input_data, model):
    model.eval()
    input_data = torch.from_numpy(input_data).float()
    logits = model(input_data).detach().numpy()
    predicted_classes = (logits > 0)*1
    odds = np.exp(logits)
    predicted_probs = odds / (1 + odds)
    print('predictions made')
    return json.dumps({'predictions': predicted_classes.tolist(),'probs': predicted_probs.tolist()})

###################################
### SAGEMAKER MODEL INPUT FUNCTION 
################################### 

def input_fn(input_data, content_type='application/x-npy'):
    input_data = io.BytesIO(input_data)
    input_data = np.load(input_data)
    print('data loaded')
    return input_data

###################################
### SAGEMAKER MODEL OUTPUT FUNCTION 
################################### 

def output_fn(prediction_output, accept='application/jsonlines'):
    return prediction_output, accept