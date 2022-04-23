import argparse
import pickle
import os
import itertools

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn import preprocessing


from utils.helper import get_classification_report

def save_model(classifier, model_path, label_encoder=None, save_dir='models'):

    model = {
        'classifier': classifier        
    }

    if label_encoder:
        model['label_encoder'] = label_encoder

    os.makedirs(save_dir, exist_ok=True)
    with open(model_path, "wb") as fout:
        pickle.dump(model, fout)


def train(data_path, baseline_strategy=None, model_path=None):
    """Train baseline tagger.

    Args:
        data_path (str): path to word embeddings training data.

    Returns:
        Any: a trained model.
    """
    print('> Training model on dataset', data_path)
    
    # This loads a dict of TESTING, TRAINING, VALIDATION keys and values as a nested list of
    # 0 as embeddings and 1 as labels (co-indexed, equal length)
    with open(data_path, "rb") as fin:
        corpus = pickle.load(fin)

    # process data
    label_encoder = preprocessing.LabelEncoder()  # labels need to be ints not strings
    all_labels = list(itertools.chain(*[corpus[split][1] for split in ['TRAINING', 'VALIDATION']]))
    label_encoder.fit(all_labels)

    train_data, train_labels_ = corpus["TRAINING"]  # the _ is the spacy convention for the string representation (rather than int/float)
    train_labels = label_encoder.transform(train_labels_)  # transform strings to ints

    if baseline_strategy:
        classifier = DummyClassifier(strategy=baseline_strategy)
    else:
        classifier = LogisticRegression(multi_class="multinomial", max_iter=500)
        
        print("> Training classifier with params:")
        print(classifier.get_params())

    if 'cupy' in str(type(train_data)):
        train_data = train_data.get()
            
    classifier.fit(train_data, train_labels)
    
    if model_path:
        print("Saving classifier to {}".format(model_path))
        save_model(classifier, model_path, label_encoder=label_encoder)
        
    print(f'> Finished training on {len(train_data)} samples')
    
    val_data, val_labels_ = corpus["VALIDATION"]

    if 'cupy' in str(type(val_data)):
        val_data = val_data.get()

    predictions = classifier.predict(val_data)
    predictions_ = label_encoder.inverse_transform(predictions)  # inverse transform to strings for printing

    print("Stats on Validation Set")
    report = get_classification_report(val_labels_, predictions_)
    print(report)
    
    return classifier, label_encoder


def evaluate(model_path, data_path, split='TESTING'):
    """Evaluate model performance on a dataset

    Args:
        model (Any): [description]
        data_path (str): path to dataset in JSON lines format
    """
    
    print('> Loading model from', model_path)

    with open(model_path, "rb") as fin:
        model = pickle.load(fin)
        
    print('> Evaluating model on dataset', data_path)

    with open(data_path, "rb") as fin:
        data, labels_ = pickle.load(fin)[split]

    predictions = model['classifier'].predict(data)
    predictions_ = model['label_encoder'].inverse_transform(predictions)  # inverse transform to strings for printing
        
    print()
    print("Full classification report:")
    report = get_classification_report(labels_, predictions_, zero_division=0, digits=3)
    print(report)

    print("\nClassification report for top tag classes (PLEASE REPORT THIS ONE):")
    report = get_classification_report(labels_, predictions_, zero_division=0, digits=3, top_k=6)
    print(report)
    

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('task', choices=['train', 'test'], help='train or evaluate the model on a test set')
    p.add_argument('--embeddings_path', help='path to embeddings file', required=True)
    p.add_argument('--model_path', help='path to load/save the model')
    p.add_argument('--baseline_strategy', choices=["most_frequent", "uniform", "stratified"], 
                   help='strategy for "dummy" classifier')
    args = p.parse_args()

    if args.task == 'train':
        train(args.embeddings_path, args.baseline_strategy, args.model_path)

    elif args.task == 'test':
        if args.model_path:
            evaluate(args.model_path, args.embeddings_path)
        else:
            raise ValueError('For evaluation, you need to specify the model_path')
