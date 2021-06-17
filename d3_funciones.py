# Librerías básicas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Métodos asociados a regularización
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# Especificaciones
warnings.filterwarnings(action= 'ignore')
plt.style.use('seaborn-whitegrid')


def palabras_frecuentes(palabras, conteo, cantidad):
    '''
    Retorna un dataframe con las palabras más fecuentes del análisis y un gráfico con el número las palabras más frecuentes del análisis, fijadas por el usuario.
    ------
    Elementos:
    palabras: Array de palabras analizadas mediante CountVectorizer.
    conteo: Array de frecuencias obtenidas mediante CountVectorizer.
    '''
    df_palabras = pd.DataFrame({'Palabras': palabras,
                                'Conteo palabras': conteo})
    df_palabras_orden = df_palabras.sort_values(by=['Conteo palabras'], ascending=False)
    df_palabras_orden.index= range(len(palabras))
    return df_palabras_orden, cantidad_frecuentes(df_palabras_orden, cantidad)

def cantidad_frecuentes(dataframe, numero):
    '''
    Retorna un gráfico con el número las palabras más fecuentes del análisis, fijadas por el usuario.
    ------
    Elementos:
    dataframe: Dataframe a analizar.
    numero: Cantidad de palabras más frecuentes a graficar.
    '''
    palabras, conteo = [],[]
    for col in dataframe:
        i = 0
        for i in range(numero):
            if (col == 'Palabras'):
                palabras.append(dataframe[col][i])
                i += 1
                if i == numero:
                    break
            else:
                conteo.append(dataframe[col][i])
                i += 1
                if i == numero:
                    break
    plt.figure(figsize=(12,10))
    plt.plot(palabras, conteo)
    plt.xticks(rotation=90)
    plt.title(f'{numero} palabras más frecuentes encontradas en las canciones')

def canciones_genero(dataframe, colestilos, estilo, letras, cantidad):
    '''
    Retorna un gráfico con las palabras más frecuentes de un determinado estilo musical.
    ---------
    Elementos:
    - dataframe: Dataframe a analizar.
    - colestilos: (Str) Columna en la cual se encuentran los estilos a analizar.
    - estilo: (Str) Nombre del estilo a analizar.
    - letras: (Str) Columna en la cual se encuentran las letras a analizar.
    - cantidad: (Int) Cantidad de palabras a analizar por estilo musical.
    '''
    palabras, conteo = [],[]
    df_analisis = dataframe[dataframe[colestilos] == estilo]
    count_vectorizer = CountVectorizer(stop_words='english')
    count_vectorizer_fit = count_vectorizer.fit_transform(df_analisis[letras])
    words = count_vectorizer.get_feature_names()
    words_freq = count_vectorizer_fit.toarray().sum(axis=0)
    df_palabras = pd.DataFrame({'Palabras': words,
                                'Conteo palabras': words_freq})
    df_palabras_orden = df_palabras.sort_values(by=['Conteo palabras'], ascending=False)
    df_palabras_orden.index= range(len(words))
    for col in df_palabras_orden:
        i = 0
        for i in range(cantidad):
            if (col == 'Palabras'):
                palabras.append(df_palabras_orden[col][i])
                i += 1
                if i == cantidad:
                    break
            else:
                conteo.append(df_palabras_orden[col][i])
                i += 1
                if i == cantidad:
                    break
    plt.figure(figsize=(12,10))
    plt.plot(palabras, conteo)
    plt.xticks(rotation=90)
    plt.title(f'{cantidad} palabras más frecuentes encontradas en el género {estilo}')

def palabras_estilo_frecuencia(dataframe, colestilo, estilo, letra, cantidad):
    count_vectorizer = CountVectorizer(stop_words='english')
    count_vectorizer_fit = count_vectorizer.fit_transform(dataframe[dataframe[colestilo]==estilo][letra])
    words = count_vectorizer.get_feature_names()
    words_freq = count_vectorizer_fit.toarray().sum(axis=0)
    return palabras_frecuentes_genero(words, words_freq, cantidad, estilo)

def palabras_frecuentes_genero(palabras, conteo, cantidad, estilo):
    '''
    Retorna un dataframe con las palabras más fecuentes del análisis y un gráfico con el número las palabras más frecuentes del análisis, fijadas por el usuario.
    ------
    Elementos:
    palabras: Array de palabras analizadas mediante CountVectorizer.
    conteo: Array de frecuencias obtenidas mediante CountVectorizer.
    '''
    df_palabras = pd.DataFrame({'Palabras': palabras,
                                'Conteo palabras': conteo})
    df_palabras_orden = df_palabras.sort_values(by=['Conteo palabras'], ascending=False)
    df_palabras_orden.index= range(len(palabras))
    return df_palabras_orden, cantidad_frecuentes_genero(df_palabras_orden, cantidad, estilo)

def cantidad_frecuentes_genero(dataframe, numero, estilo):
    '''
    Retorna un gráfico con el número las palabras más fecuentes del análisis, fijadas por el usuario.
    ------
    Elementos:
    dataframe: Dataframe a analizar.
    numero: Cantidad de palabras más frecuentes a graficar.
    '''
    palabras, conteo = [],[]
    for col in dataframe:
        i = 0
        for i in range(numero):
            if (col == 'Palabras'):
                palabras.append(dataframe[col][i])
                i += 1
                if i == numero:
                    break
            else:
                conteo.append(dataframe[col][i])
                i += 1
                if i == numero:
                    break
    plt.figure(figsize=(14,10))
    plt.plot(palabras, conteo)
    plt.xticks(rotation=90)
    plt.title(f'{numero} palabras más frecuentes encontradas en las canciones del género {estilo}')

def compare_priors(X_train, X_test, y_train, y_test, prior):
    """TODO: Docstring for compare_priors.

    :prior: TODO
    :returns: TODO

    """
    tmp_clf = BernoulliNB(class_prior=prior)
    tmp_clf.fit(X_train, y_train)
    tmp_class = tmp_clf.predict(X_test)
    tmp_pr = tmp_clf.predict_proba(X_test)[:, 1]
    tmp_acc = accuracy_score(y_test, tmp_class).round(3)
    tmp_rec = recall_score(y_test, tmp_class).round(3)
    tmp_prec = precision_score(y_test, tmp_class).round(3)
    tmp_f1 = f1_score(y_test, tmp_class).round(3)
    tmp_auc = roc_auc_score(y_test, tmp_pr).round(3)
    print("A priori: {0}\nAccuracy: {1}\nRecall: {2}\nPrecision: {3}\nF1: {4}\nAUC: {5}\n".format(prior, tmp_acc, tmp_rec, tmp_prec, tmp_f1, tmp_auc))

def deaggregate_statistics(dataframe):
    """Given a frequency, multiply attributes combination row-wise and generate a new dataframe

    :dataframe: TODO
    :returns: TODO

    """
    final_band = []
    final_genus = []
    final_single = []
    final_lyric = []

    for _, row_serie in dataframe.iterrows():
        for _ in range(1, row_serie[4] + 1):
            final_band.append(row_serie[0])
            final_genus.append(row_serie[1])
            final_single.append(row_serie[2])
            final_lyric.append(row_serie[3])

    return pd.DataFrame({
        '0': final_band,
        '1': final_genus,
        '2': final_single,
        '3': final_lyric
    })