import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import stop_words
import nltk
from nltk.stem import  SnowballStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_ru = set(stopwords.words('russian'))
stop =  stop_ru | set(stop_words.get_stop_words('ru')) -set('год')

stemmer_ru = SnowballStemmer('russian')
stemmer_eng = SnowballStemmer('english')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

def data_pretransform(df, type='tf-idf', maxlen = 150, max_features = 4000):
    df = df[(df.text.notnull())]
    df = df.reset_index(drop = True)

    def string_transform(string):
        string = string.lower().split()
        string = [stemmer_eng.stem(stemmer_ru.stem(i)) for i in string if i not in stop]

        return ' '.join(string)
    
    df['text'] = df['text'].apply(string_transform)

    if type=='tf-idf':
        text_tf_idf = TfidfVectorizer(min_df=0.01, ngram_range = (1,2))
        text_data = text_tf_idf.fit_transform(df['text'].astype(str))
        return df, text_data
    
    elif type=='word2vec':
        
        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(list(['text']))
        text_data = pad_sequences(tokenizer.texts_to_sequences(df['text']), maxlen=maxlen)
        return df, text_data
    
    
    
import seaborn as sns
def plot_confusion_matrix(y_test, prediction, class_names, figsize = (10,7), fontsize=14):

    
    matrix = pd.DataFrame(confusion_matrix(y_test, prediction))
    m=(matrix/matrix.sum()*10**2).fillna(0).astype(int)
    confusion_matrix1 = m.values
    
    
    df_cm = pd.DataFrame(
        confusion_matrix1, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
