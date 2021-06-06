import argparse
import gzip
import json
from typing import Iterable
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import dill
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

import os


class Model:
    def __init__(self):
        self.vec = CountVectorizer()
        self.tfidf = TfidfTransformer()
        self.model = MultinomialNB()

   


    def train_with_hp(self, train_data: Iterable[dict]): 
        x_train=[x['text'] for x in train_data]
        y_train=[x['label'] for x in train_data]
        
        clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
        ])

        parameters = {
        'clf__alpha': np.linspace(0.5, 1.5, 6),
        'clf__fit_prior' : [True, False],
        'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2')
        }
        gs_clf = RandomizedSearchCV(clf, parameters)
        gs_clf = gs_clf.fit(x_train,y_train)
        print("Best score accurracy = %.3f%%" %((gs_clf.best_score_)*100.0))
        print("Best parameters are : ")
        print(gs_clf.best_params_)

    def train(self, train_data: Iterable[dict]):       
 
        counts = self.vec.fit_transform([x['text'] for x in train_data])
        tfidf = self.tfidf.fit_transform(counts)
        
        self.model.fit(tfidf, [x['label'] for x in train_data])

    def predict(self, data: Iterable[dict]):
        return self.model.predict(
            self.tfidf.transform(self.vec.transform([x['text'] for x in data]))
        )
    
    def get_accuracy(self,prediction,data: Iterable[dict]):
        return accuracy_score(prediction, [x['label'] for x in data])
    
    def save(self, filename):
         with open(filename, 'wb') as f:
            dill.dump(self, f,recurse=True)    
  
def load_dataset(path):
    data = []
    with gzip.open(path, "rb") as f_in:
        for line in f_in:
            data.append(json.loads(line))
    return data


def main(args):
    train_data = load_dataset(args.train)
    model = Model()
    model.train(train_data)
    model.train_with_hp(train_data)
    test_data = load_dataset(args.test)
    prediction=model.predict(test_data)
   
    print(f"\nPredicted label: {prediction}")   
    print('accuracy %s' % model.get_accuracy(prediction,data=test_data))
    print(classification_report([x['label'] for x in test_data], prediction))
    
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'joblib_cl_model.pkl')
    model.save(filename)
  

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=str,
        default="train.jsonl.gz",
        help="path to training data (.jsonl.gz file)",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="test.jsonl.gz",
        help="path to test data (.jsonl.gz file)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
