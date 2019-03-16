"""
Helper Functions for code that is not shown in the notebooks
"""

import numpy as np
import pandas as pd
import re
import os.path
import math
from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn.cross_validation import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from scipy import sparse
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
#from skmultilearn.problem_transform import LabelPowerset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer



def accuracy(y_actual, y_predict):
    '''
    Returns classification report as a data frame
    '''
    result = pd.DataFrame(columns=['Precision', 'Recall', 'F1-Score', 'Support'])
    tp, fp, fn, total = 0, 0, 0, 0
    if not isinstance(y_predict, pd.DataFrame):
        y_predict = pd.DataFrame(y_predict, columns=y_actual.columns)
    for col in y_actual.columns:
        support = y_actual[col].sum()
        tp = ((y_actual[col]==1) & (y_predict[col]==1)).sum()
        fp = ((y_actual[col]==0) & (y_predict[col]==1)).sum()
        fn = ((y_actual[col]==1) & (y_predict[col]==0)).sum()
        
        precision = 0 if (tp+fp==0) else tp/(tp+fp)
        recall = 0 if (tp+fn==0) else tp/(tp+fn)
        f1_score = 0 if (precision==0 and recall==0) else 2*precision*recall/(precision+recall)
        
        result.loc[col] = [precision, recall, f1_score, support]
    
    avg_precision = (result['Precision']*result['Support']).sum()/result['Support'].sum()
    avg_recall = (result['Recall']*result['Support']).sum()/result['Support'].sum()
    avg_f1_score = (result['F1-Score']*result['Support']).sum()/result['Support'].sum()
    result.loc['Avg/Total'] = [avg_precision, avg_recall, avg_f1_score, result['Support'].sum()]
    
    return round(result, 2)


def overall_f1_score_v1(y_actual, y_predict):
    '''
    Overall F1 Score. Used as our final evaluation metric 
    v1: Used for Binary Relevance. 
    y_actual and y_predict have dimensions n_rows x num_genres
    '''
    num_genres = y_actual.shape[1]
    tp, fp, fn = 0, 0, 0
    for idx in range(num_genres):
        tp+=((y_actual.iloc[:,idx]==1) & (y_predict[:,idx]==1)).sum()
        fp+=((y_actual.iloc[:,idx]==0) & (y_predict[:,idx]==1)).sum()
        fn+=((y_actual.iloc[:,idx]==1) & (y_predict[:,idx]==0)).sum()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*precision*recall/(precision+recall)
    return f1_score

def overall_f1_score_v2(y_actual, y_predict, class_to_genre_map=None):
    '''
    Overall F1 Score. Used as our final evaluation metric 
    v2: Used for Label Powerset
    Here y_actual and y_predict have dimensions of n_rows x 1. 
    Prediction for each row would be a label whose mapping to respective genre
    combination is provided in class_to_genre_map
    '''
    num_class, num_genres = class_to_genre_map.shape[0], class_to_genre_map.shape[1]
    
    y_actual_matrix = np.empty((y_actual.shape[0], num_genres))
    y_predict_matrix = np.empty((y_actual.shape[0], num_genres))
    
    
    for idx in range(num_class):
        if idx in y_predict:
            y_predict_matrix[y_predict==idx,:] = class_to_genre_map.loc[idx,:].values
        if idx in y_actual:
            y_actual_matrix[y_actual==idx,:] = class_to_genre_map.loc[idx,:].values 
    
    tp, fp, fn = 0, 0, 0
    for idx in range(num_genres):
        tp+=((y_actual_matrix[:,idx]==1) & (y_predict_matrix[:,idx]==1)).sum()
        fp+=((y_actual_matrix[:,idx]==0) & (y_predict_matrix[:,idx]==1)).sum()
        fn+=((y_actual_matrix[:,idx]==1) & (y_predict_matrix[:,idx]==0)).sum()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*precision*recall/(precision+recall)
    return f1_score


def get_features_tfidf(pipeline, labels, num_features=10, prob_only=False):
    '''
    Returns the top features (words) and the probability weight of the genres listed in 'labels'
    If prob_only = True, it only returns the probability
    '''
    category_columns = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
       'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Game-Show',
       'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
       'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Talk-Show',
       'Thriller', 'War', 'Western']
    tfidf_vectorizer = pipeline.get_params()['tfidf']
    clf = pipeline.get_params()['clf']
    words = np.array(tfidf_vectorizer.get_feature_names())
    x=sparse.identity(words.shape[0])
    if num_features<=0:
        num_features = words.shape[0]
    probs = clf.predict_proba(x)
    feature_id = ['P{0}'.format(i) for i in range(1, num_features+1)]
    df_good_features = pd.DataFrame(columns=feature_id)
    for col in labels:
        ind = np.argsort(probs[:,list(category_columns).index(col)])
        
        good_prob = probs[ind[-num_features:], list(category_columns).index(col)]
        if prob_only:
            good_list = good_prob
        else:
            good_words = words[ind[-num_features:]]
            good_list = ['{0} (p={1:.2f})'.format(w, p) for w, p in zip(good_words, good_prob)]
        
        df_good_features.loc[col,:] = good_list[::-1]
    return df_good_features


def get_prob_thresh(mydata, thresh_sel=1, thresh_offset=0):
    '''
    The probability threshold to be used for making classification decisions
    thresh_sel = 1 : Default 0.5 probability threshold
    thresh_sel = 2 : max(0.5, Fraction of genre occurence + thresh_offset)
    '''
    num_genres = mydata.shape[1]
    prob_thresh=[]
    if thresh_sel==1:
        prob_thresh = [0.5] * num_genres
    elif thresh_sel==2:
        sum_genre = mydata.sum()
        prob_thresh = (sum_genre/mydata.shape[0]+thresh_offset).clip(upper=0.5)
    return prob_thresh



def multi_label_predict(clf, X_test, prob_thresh):
    '''
    Multi-label prediction  based on probability threshold. 
    Prediction is made based on Binary Relevance where each genre has a separate classifier
    '''
    category_columns = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
       'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Game-Show',
       'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
       'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Talk-Show',
       'Thriller', 'War', 'Western']
    
    y_pred = pd.DataFrame(columns=category_columns)
    
    prob = clf.predict_proba(X_test)
    for idx, col in enumerate(category_columns):
        y_pred[col] = prob[:,idx]>prob_thresh[idx]
    prob = pd.DataFrame(prob, columns=category_columns)
    return prob, y_pred



def multi_class_predict(clf, X_test, class_to_genre_map):
    '''
    Multi-label prediction based on Label Powerset
    Predictions made by classifiers are labels ranging from 0 to num_class
    class_to_genre_map maps the labels to genre combination
    '''
    if isinstance(test_X, pd.Series):
        y_pred = pd.DataFrame(index=X_test.index, columns=class_to_genre_map.columns)
    else:
        y_pred = pd.DataFrame(index=range(X_test.shape[0]), columns=class_to_genre_map.columns)
    
    num_class = class_to_genre_map.shape[0]
    y_class = clf.predict(X_test)
    for idx in range(num_class):
        if idx in y_class:
            y_pred.loc[y_class==idx,:] = class_to_genre_map.loc[idx,:].values
    return y_pred


def analyze_plot_genre(pipeline, plot, genre):
    '''
    For each feature (or word) in the plot, it outputs the weight for the genres listed in 'genre'
    '''
    category_columns = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
       'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Game-Show',
       'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
       'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Talk-Show',
       'Thriller', 'War', 'Western']
    
    plot = pd.Series(plot.split())
    
    tfidf_vectorizer = pipeline.get_params()['tfidf']
    clf = pipeline.get_params()['clf']
    words = np.array(tfidf_vectorizer.get_feature_names())
    x=sparse.identity(words.shape[0])
    
    probs = pipeline.predict_proba(plot)
    
    result = pd.DataFrame(columns=['word']+genre)
    result['word'] = plot
    for col in genre:
        result[col] = probs[:, list(category_columns).index(col)]
    
    return result
