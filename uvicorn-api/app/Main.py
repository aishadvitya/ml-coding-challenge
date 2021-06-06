# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 21:08:29 2021

@author: aisha
"""
# Data Handling
import pickle
import numpy as np
from pydantic import BaseModel
# Server
import uvicorn
from fastapi import FastAPI
# Modeling


app = FastAPI()

# Initialize files
clf = pickle.load(open('./app/joblib_cl_model.pkl', 'rb'))

class Data(BaseModel):
    text:str
   
        
@app.post("/prediction")
def predict(data:Data):  
    # Extract data in correct order
  
      data_dict = data.dict()  
      prediction=clf.predict([data_dict]) 
      return {'label':str(prediction)}
