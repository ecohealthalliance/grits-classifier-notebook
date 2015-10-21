import datetime
import os
import pickle
import diagnosis
from diagnosis.KeywordExtractor import *
import numpy as np
import re
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from diagnosis.utils import group_by, flatten
import warnings
import pymongo
from DataSet import fetch_datasets


def main():
    print "Setting up"

    with open('ontologies.p') as f:
        keywords = pickle.load(f)

    categories = set([
        'hm/disease',
        'biocaster/pathogens',
        'biocaster/diseases',
        'biocaster/symptoms',
        'symp/symptoms',
        'eha/symptom',
        'eha/mode of transmission',
        'eha/environmental factors',
        'eha/vector',
        'eha/occupation',
        'eha/control measures',
        'eha/description of infected',
        'eha/disease category',
        'eha/host',
        'eha/host use',
        'eha/symptom',
        'eha/disease',
        'eha/location', 
        'eha/transmission',
        'eha/zoonotic type',
        'eha/risk',
        'wordnet/season',
        'wordnet/climate',
        'wordnet/pathogens',
        'wordnet/hosts',
        'wordnet/mod/severe',
        'wordnet/mod/painful',
        'wordnet/mod/large',
        'wordnet/mod/rare',
        'doid/has_symptom',
        'doid/symptoms',
        'doid/transmitted_by',
        'doid/located_in',
        'doid/diseases',
        'doid/results_in',
        'doid/has_material_basis_in',
        'usgs/terrain'
    ])

    keyword_array = [
        keyword_obj for keyword_obj in keywords
        if keyword_obj['category'] in categories
    ]

    feature_extractor = Pipeline([
        ('kwext', KeywordExtractor(keyword_array)),
        ('link', LinkedKeywordAdder(keyword_array)),
        ('limit', LimitCounts(1)),
    ])

    print "Fetching datasets"
    time_offset_test_set, mixed_test_set, training_set = fetch_datasets()

    print "Setting up vectorizers and extractors"
    time_offset_test_set.feature_extractor =mixed_test_set.feature_extractor =training_set.feature_extractor = feature_extractor
    my_dict_vectorizer = DictVectorizer(sparse=False).fit(training_set.get_feature_dicts())
    time_offset_test_set.dict_vectorizer = mixed_test_set.dict_vectorizer = training_set.dict_vectorizer = my_dict_vectorizer

    print "Removing zero feature vectors"
    time_offset_test_set.remove_zero_feature_vectors()
    mixed_test_set.remove_zero_feature_vectors()
    training_set.remove_zero_feature_vectors()

    feature_array = np.array(training_set.get_feature_vectors())
    label_array = np.array(training_set.get_labels())

    classifiers = [
        (OneVsRestClassifier(LogisticRegression(), n_jobs=-1), "OneVsRest(Logistic Regression)"),
        (OneVsRestClassifier(SVC(), n_jobs=-1), "OneVsRest(SVC)"),
        (DecisionTreeClassifier(), "Decision Tree Classifier"),
        (AdaBoost(DecisionTreeClassifier()), "AdaBoost(Decision Tree Classifier)")
    ]

    for (my_classifier, classifier_label) in classifiers:

        print("Fitting classifier: " + classifier_label)
        my_classifier.fit(feature_array, label_array)
        print("Classifier fit")

        print("Testing:")

        for data_set, ds_label, print_label_breakdown in [
            (training_set, "Training set", False),
            (time_offset_test_set, "Time offset set", True),
            (mixed_test_set, "Mixed test set", False),
        ]:
            y_pred_flat = my_classifier.predict(data_set.get_feature_vectors())
            y_pred = [[y] for y in y_pred_flat]
            print (ds_label + "\n"
                "precision: %s recall: %s f-score: %s") %\
                sklearn.metrics.precision_recall_fscore_support(
                    data_set.get_labels(),
                    y_pred,
                    average='micro')[0:3]

if __name__ == "__main__":
    main()
