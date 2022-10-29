import json
from collections import namedtuple

import numpy as np
import pandas as pd

def input_handler(data, context):
    payload = pd.read_csv(data).iloc[:,:-1].to_numpy()
    #instance = [{'dataset': payload.tolist()}]
    return json.dumps({'instances': payload.tolist()})

def output_handler(data, context):
    if data.status_code != 200:
        raise Exception(data.content.decode('utf-8'))
    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type