from fastapi import FastAPI
import pickle
# import pandas as pd
import numpy as np
from pydantic import BaseModel

model_file = pickle.load(open('model/logistic_model.pkl', 'rb'))

app = FastAPI(
    title="Breast Cancer prediction applicaton",
    version="0.0.1"
)

class FeatureData(BaseModel):
    features: str

"""
req should be like this- 
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "features": "-0.23711093,-0.4976419,0.61365274,-0.49813131,-0.53102815,-0.57694824,-0.17494424,-0.36215622,-0.284859,   0.43345165, 0.17818232,-0.36844966,0.55310406,-0.31671104,-0.40524636, 0.04025752,-0.03795529,-0.18043065,0.16478901,-0.12170969, 0.23079329,-0.50044002, 0.81940367,-0.46922838,-0.53308833,-0.04910117,-0.04160193,-0.14913653, 0.09681787, 0.10617647,0.49035329"
}'
"""

@app.post('/predict')
def predict(features_data: FeatureData):
    features_list = features_data.features.split(',')
    np_features = np.asarray(features_list, dtype=np.float32)
    pred = model_file.predict(np_features.reshape(1,-1))
    output = ['Cancer' if pred[0] == 1 else 'Not Cancer']

    return output
