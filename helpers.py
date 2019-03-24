#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from scipy import sparse
import seaborn as sns

def identify_words_associated(model, counter, words_per_topic):
    """TODO: Docstring for identify_words_associated.

    :model: TODO
    :counter: TODO
    :words_per_topic: TODO
    :returns: TODO

    """
    # para cada componente del modelo (parámetros variacionales de la distribución de palabras en tópicos)
    for idx, name in enumerate(model.components_):
        # asociamos las W palabras ordenadas por tópico
        words_on_topic = " ".join(
            [counter.get_feature_names()[i] for i in name.argsort()[:-words_per_topic - 1: -1] ]
        )
        # retorna una lista de tópicos
        print(f"Tópico: {idx}\t\t\n{words_on_topic}")

def identify_most_likely_topic(model, counter, sparse_feats, df):
    """TODO: Docstring for identify_most_likely_topic.

    :model: TODO
    :counter: TODO
    :sparse_feats: TODO
    :df: TODO
    :returns: TODO

    """
    # en base al modelo, inferimos las representaciones
    infer_topics = model.transform(sparse_feats)
    # generamos etiquetas de columnas para una lista de tópicos
    topics = list(
        map(lambda x: f"Tópico {x}", range(1, model.n_components + 1))
    )
    # una lista de canciones
    docsnames = list(
        map(lambda x: f"Canción {x}", range(df.shape[0]))
    )
    # organizamos los tópicos inferidos con sus etiquetas asociadas
    docs_topics = pd.DataFrame(
        np.round(infer_topics, 3),
        columns = topics,
        index=df.index
    )
    # concatenamos la mezcla de tópicos a la información
    concatenate_dataframe = pd.concat(
        [df, docs_topics],
        axis=1
    )
    # definimos de forma explícita el maximo a posteriori
    concatenate_dataframe['argmax_pr'] = np.argmax(docs_topics.values, axis=1)+1
    # retornamos un dataframe
    return concatenate_dataframe

def report_artist_topic(infered_df, artist, plot_means = True, topic_label=None):
    """TODO: Docstring for report_artist_topic.

    :infered_df: TODO
    :artist: TODO
    :plot_means: TODO
    :returns: TODO

    """
    # en base al objeto retornado con identify_most_likely_topic, subseteamos por artista
    tmp_artist_topics = infered_df.query(f"artist == '{artist}'")
    # extraemos solo las columnas identificadas como topico
    topic_labels = [i if i.startswith('Tópico') else None for i in tmp_artist_topics]
    # eliminamos las que no sean tópicos
    topic_labels = list(set(topic_labels).difference(set([None])))
    # subseteamos columnas
    tmp_artist_topics = tmp_artist_topics.loc[:, topic_labels]
    # calculamos la media a nivel de columna
    tmp_artist_means = tmp_artist_topics.apply(np.mean, axis=0)

    if topic_label is not None:
        tmp_artist_topics.columns=topic_label

    for index, (colname, serie) in enumerate(tmp_artist_topics.iteritems()):
        plt.subplot(len(topic_labels), 1, index + 1)
        sns.distplot(serie)
        sns.despine()
        if plot_means is True:
            plt.axvline(tmp_artist_means[index])

        plt.title(colname)
        plt.xlim(0, 1)
        plt.xlabel('')
        plt.tight_layout()

    return tmp_artist_means
