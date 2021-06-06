import connexion
import six

from openapi_server.models.api_request import ApiRequest  # noqa: E501
from openapi_server.models.api_response import ApiResponse  # noqa: E501
from openapi_server import util
import pickle
import os
from openapi_server.models.model import Model

def prediction(body):  # noqa: E501
    """prediction

     # noqa: E501

    :param body: The document to perform a prediction on
    :type body: dict | bytes

    :rtype: ApiResponse
    """
    if connexion.request.is_json:
        data_dict = connexion.request.get_json()  # noqa: E501
       # data_dict=body.dict()
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'joblib_cl_model.pkl') 
        print('Filename',filename)
        with open(filename, 'rb') as f:
          saved_model = pickle.load(f)
        predicted=saved_model.predict([data_dict])
    return {'label':str(predicted)}
