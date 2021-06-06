# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 23:13:20 2021

@author: aisha
"""

import requests
import json

def test_post_headers_body_json():
    url = 'http://localhost:8080/prediction'
    
    # Additional headers.
    headers = {'Content-Type': 'application/json' } 

    # Body
    payload = {'text':'some text about apollo'}
    
    # convert dict to json string by json.dumps() for body data. 
    resp = requests.post(url, headers=headers, data=json.dumps(payload,indent=4))       
    
    # Validate response headers and body contents, e.g. status code.
    assert resp.status_code == 200
    resp_body = resp.json()
    assert resp_body['label']=="['cryptography']"    
    # print response full body as text
    print(resp.text)