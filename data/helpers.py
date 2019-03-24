#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: helpers.py
Author: Ignacio Soto Zamorano
Email: ignacio[dot]soto[dot]z[at]gmail[dot]com
Github: https://github.com/ignaciosotoz
Description: Ancilliary Files - LatentDirichletAllocation example - Machine Learning ADL
"""


import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from scipy import sparse
from typing import TypeVar


LDA = TypeVar('sklearn.decomposition.LatentDirichletAllocation')
CounterVec = TypeVar('sklearn.feature_extraction.text.CountVectorizer')
SparseMatCSR = TypeVar('scipy.sparse.csr_matrix')
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')

def identify_words_associated(model: LDA, counter: CounterVec, words_per_topic:int =15) -> str:
    """identify_words_associated - Given a fitted model and an associated vectorized corpus, extract N words given K inferred topics.

    :model: A sklearn.decomposition.LatentDirichletAllocation fitted model
    :counter: A sklearn.feature_extraction.text.CountVectorizer vectorized corpus
    :words_per_topic: An integer defining the top N words associated with a K topic.
    :returns: Printed list

    """
    if isinstance(counter, CountVectorizer) is True:
        for topic_id, topic_name in enumerate(model.components_):
            words_on_topic = "\t\n".join([
                counter.get_feature_names()[i] for i in topic_name.argsort()[:-words_per_topic - 1: -1]
            ])
            print(f"Tópico: {topic_id}\n {words_on_topic}")
    else:
        raise TypeError('counter param is not a CountVectorizer class')

def identify_most_likely_topic(model: LDA, counter: CounterVec, sparse_feats: SparseMatCSR, df: PandasDataFrame) -> PandasDataFrame:
    """identify_most_likely_topic - given a trained LDA model, a vectorized corpus and a dataframe, insert class probabilities and its highest pr topic

    :model: A sklearn.decomposition.LatentDirichletAllocation object
    :counter: A sklearn.feature_extraction.text.CountVectorizer object
    :sparse_feats: A scipy.sparse.csr_matrix object created with sklearn.feature_extraction.text.CountVectorizer.transform
    :returns: A pd.DataFrame object with imputed columns.

    """
    lda_validate = isinstance(model, LatentDirichletAllocation)
    counter_validate = isinstance(counter, CountVectorizer)
    csr_validate = isinstance(sparse_feats, sparse.csr_matrix)

    if lda_validate and counter_validate and csr_validate:
        infer_topics = model.transform(sparse_feats)
        topics = list(
            map(lambda x: f"Topic {x}", range(1, model.n_components + 1))
        )
        document_name = list(
            map(lambda x: f"Song {x}", range(df.shape[0]))
        )
        docs_topics = pd.DataFrame(np.round(infer_topics, 3),
                                   columns=topics, index=df.index)
        concatenate_pd = pd.concat([df, docs_topics], axis=1)
        concatenate_pd['highest_topic_pr'] = np.argmax(docs_topics.values, axis=1) + 1
        return concatenate_pd
    else:

        if lda_validate is False:
            raise TypeError('Model is not LatentDirichletAllocation')
        if counter_validate is False:
            raise TypeError('Counter is not CountVectorizer')
        if csr_validate is False:
            raise TypeError('sparse_feats is not scipy.sparse.csr_matrix')

def correlate_lda_topics(df: PandasDataFrame):
    """TODO: Docstring for correlate_lda_topics.

    :df: TODO
    :returns: TODO

    """
    tmp_df = df.filter(regex = 'Topic*').corr()
    sns.heatmap(tmp_df, annot=True, cmap='Greys')