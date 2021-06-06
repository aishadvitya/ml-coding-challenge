## Challenge: 
You are provided with the following:

* A small text dataset, with training data and test data in the files [train.jsonl.gz](./train.jsonl.gz). and [test.jsonl.gz](./test.jsonl.gz) respectively.
    * The text is in English, and is from a collection of posts to newsgroups on various topics.
    * There are 6 labels: `"space", "electronics", "cryptography", "politics", "hockey", "baseball"`.
    * Each line contains a single example encoded as a JSON object: `{"text": "foo content", "label": "foo label"}`.

* A text classifier created using scikit-learn, in the file `model.py`.
    * You can install the necessary dependencies for the model into your local environment by running `pip install -r requirements.txt`.
    * When executed, this Python module will train a classifier and output a prediction for a single test example to standard output.


_1. Provide an appropriate evaluation of the model performance on the test data._
   Check for the following code in model.py
    
    def get_accuracy(self,prediction,data: Iterable[dict]):
        return accuracy_score(prediction, [x['label'] for x in data])
        
_2. Implement a way to persist the trained model to local disk._
      This is done using dill. Dill helps to persist the class which also has a standalone __main__. This can also be accomplished by pickle or joblib with level setting -1(highest level for nested classses)
      
_3. Implement an API according to the open api specification._
       can run the following command to generate openapi-server code
       
       pip3 install openapi-generator
       openapi-generator generate -i prediction-openapi.yaml -g python-flask -o codegen_server/
       
4. Create a web service (in Python 3) to serve the persisted model.
    There are two rest services:
    
    **Uviciorn**: that internally uses FASTAI which uses OpenAPI in the folder uvicorn-api. Can be run with :
    pip3 install uvicorn 
    uvicorn Main:app
    This starts the service in port 8000
    
    **Open API Server** : This was from the previously generated server stub modified for our service. Can be run with :
    python -m openapi_server
     
6. Deploy the model locally.
   Run scripts above to deploy to ports 8000 for Uvicorn & 8080 for openAPI . Of course, the ports can be changed with --p parameter . 
   
   
8. Create a container with your solution that can be run on Kubernetes.

   The Docker files have been added in the respective folders along with requirements in requirements.txt
   you can create the docker image using the following commands:
   **uvicorn**:
       
        cd uvicorn-api
        docker build -t uvicorn-image:latest .
        docker run -d -p 8000:8000 uvicorn-image
        
   **open-api**
     
       cd codegen-server
       docker build -t codegen-image:latest . 
       docker run -d -p 8080:8080 codegen-image
10. Provide some sample curl commands or a [Postman](https://www.postman.com/) collection.
     curl --location --request POST 'http://localhost:8080/prediction?body=This%20is%20a%20sample%20text%20for%20prediction%20testing%20something%20about%20apollo%2011%20' \
   --header 'Content-Type: application/json' \
--data-raw '{"text": "some text about apollo"}'

sample reponse:
{
    "label": "['cryptography']"
}

12. *Stretch Goal 1* - Suggest and/or implement improvements to the model.
       Added new method _train_with_hp_ in model.py for hyperparameter tuning to multinomialNB , Ngram vectr, use_idf  using RandomizedSearchCV . 
       Results are as follows:
       **_After hyperparameter tuning_**
       Best score accurracy = 94.885%
       Best parameters are : 
       {'vect__ngram_range': (1, 1), 'tfidf__use_idf': True, 'tfidf__norm': 'l2', 'clf__fit_prior': True, 'clf__alpha': 0.7}

       Predicted label: ['space' 'space' 'space' ... 'baseball' 'hockey' 'baseball']
       **_Previous model_**
       accuracy 0.8702490170380078
       
14. *Stretch Goal 2* - Testing of the API before deployment.
       install pytest using 
       >>> pip3 install -U requests Flask pytest pytest-html
       >>> cd test-api
       >>> pytest 
       =========================================================================================================== test session starts ============================================================================================================
platform win32 -- Python 3.8.5, pytest-6.2.4, py-1.9.0, pluggy-0.13.1
rootdir: D:\ML\ml-challenge\ml-challenge-main\test-api
plugins: html-3.1.1, metadata-1.11.0
collected 1 item

test_8080.py .                                                                                                                                                                                                                        [100%]

============================================================================================================= warnings summary =============================================================================================================
c:\users\aisha\anaconda3\lib\site-packages\pyreadline\py3k_compat.py:8
  c:\users\aisha\anaconda3\lib\site-packages\pyreadline\py3k_compat.py:8: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated since Python 3.3, and in 3.9 it will stop working
    return isinstance(x, collections.Callable)

-- Docs: https://docs.pytest.org/en/stable/warnings.html
======================================================================================================= 1 passed, 1 warning in 0.24s =======================================================================================================

  For publishing report , run  pytest -sv --html report.html

       
16. *Stretch Goal 3* - Metrics API for inspecting current metrics of the service.
You could use Prometheus or Consul 

prometheus has better integration with flask 
pip3 install prometheus-flask-exporter

from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)
metrics = PrometheusMetrics(app)
metrics.info('app_info', 'Application info', version='1.0.3')

All @app.route is tracked by default. You can use @metrics.do_not_track() if needbe . 

If you are looking for running it directly on kubernetes, you can use Consul. Consul agents can be directly used with kubernetes(using helm chart , needs helm 2 or 3)



