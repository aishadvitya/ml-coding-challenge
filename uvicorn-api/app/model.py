import argparse
import gzip
import json
from typing import Iterable
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import dill


import os


class Model:
    def __init__(self):
        self.vec = CountVectorizer()
        self.tfidf = TfidfTransformer()
        self.model = MultinomialNB()

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
    test_data = load_dataset(args.test)
    prediction=model.predict(test_data)
   
    print(f"\nPredicted label: {prediction}")
    print('accuracy %s' % model.get_accuracy(prediction,data=test_data))
    print(classification_report([x['label'] for x in test_data], prediction))
    dict_n=[{'text':"Somenonese about apollo"}]
    print([x['text']for x in dict_n])
    pred=model.predict(dict_n)
    print('In here',pred)
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
