'''
Script with examples for information retrieval.

-Mikko Lempinen
'''
import nltk
import numpy as np
import pandas as pd
import heapq
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from urllib.error import HTTPError
from urllib.request import urlopen
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

def get_Text(link):
    '''
    Scrapes meaningful text from a given html page.
    :param link: (str) URL address.
    '''
    soup = BeautifulSoup(urlopen(link),'lxml')
    soup = soup.find('div', id='mw-content-text')#.find('div',)
    Text = ''
    for item in soup.find_all('p'):#, recursive=False):
        Text += item.text.strip()
    return Text

def get_Finnish_Cities():
    '''
    Scrape Finnish Cities & Towns data from Wikipedia.
    '''
    soup = BeautifulSoup(urlopen('https://en.wikipedia.org/wiki/List_of_cities_and_towns_in_Finland'),'lxml')
    soup = soup.find('div',class_='mw-parser-output')
    Table = soup.find('table', style="text-align:right;").find('tbody')
    
    for item in Table.find_all('tr'):
        if item.find('td'):
           name = item.find('td').find('a')
           if name:
              Finnish_Cities[name['title']] = get_Text(base_link+name['href'])
           else:
               name = item.find_all('td')[1].find('a')
               Finnish_Cities[name['title']] = get_Text(base_link+name['href'])

def preProcess(doc):
    '''
    Preprocessor for a corpus. Removes stopwords and stems & lemmatizes words.
    '''
    Stopwords = list(set(nltk.corpus.stopwords.words('english')))
    stemmer = SnowballStemmer("english")
    WN_lemmatizer = WordNetLemmatizer()

    sentences = sent_tokenize(doc)
    Tokens = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [stemmer.stem(word) for word in words]
        words = [WN_lemmatizer.lemmatize(word, pos="v") for word in words]
        
        words = [word for word in words if word.isalpha() and word not in Stopwords] #get rid of numbers and Stopwords
        #words= [word for word in words if word.isalnum() and word not in Stopwords]
        Tokens.extend(words)
    return ' '.join(Tokens)

def TFIDF(corpus, preprocessor=None):
    '''
    Creates a TF-IDF vector of a given corpus.
    :arg corpus: (list) List of text documents.
    :kwarg preprocessor: (function) Preprocessing function. (Optional)
    '''
    if preprocessor:
        Tfidf = TfidfVectorizer(preprocessor=preprocessor)
    else:
        Tfidf = TfidfVectorizer()
    Tfidf.fit(corpus)
    #feature_names = Tfidf.get_feature_names_out()
    #print(Tfidf.vocabulary_)
    #X = Tfidf.transform(corpus)
    #X = Tfidf.fit_transform(corpus)
    #Tfidf = Tfidf.transform(corpus)
    return Tfidf

def BOW_model(corpus, preprocessor=None):
    '''
    Creates a CountVector of a given corpus.
    :arg corpus: (list) List of text documents.
    :kwarg preprocessor: (function) Preprocessing function. (Optional)
    '''
    if preprocessor:
        BOW = CountVectorizer(preprocessor=preprocessor)
    else:
        BOW = CountVectorizer()
        #max_df=0.8, min_df=0.2, )  in case you want to reduce the size of the dictionary, you can change the default values of max_de, min_def and other parameters
    BOW.fit(corpus)
    #X = BOW.transform(corpus)
    #X = BOW.fit_transform(corpus)
    return BOW

def evaluate_query(query, vectorizer):
    '''
    Find the closest matching documents for given query.
    :param query: (list) List of queries to evaluate.
    :param vectorizer: (sklearn.Vectorizer) A sklearn Vectorizer class.
    '''
    vectors = vectorizer.transform(list(Finnish_Cities.values())).toarray()
    Vq = vectorizer.transform(query).toarray()[0]
    
    Scores = []
    for vector in vectors:
        Scores.append(np.inner(Vq, vector))
    max_score_cities = []
    max_score_indexes = heapq.nlargest(3, range(len(Scores)), key=Scores.__getitem__)
    for i in max_score_indexes:
        max_score_cities.append(list(Finnish_Cities)[i])


    print('Document with the highest score: City of',max_score_cities[0])
    print('Document with the 2nd highest score: City of',max_score_cities[1])
    print('___________________________\n')


if __name__ == '__main__':
    #Make sure wordnet resource is downloaded
    try:
        wordnet.synonyms('car')
    except LookupError:
        nltk.download('wordnet')

    query = ['I will visit Oulu this summer and possibly Espoo.']
    Finnish_Cities = {}
    base_link = 'https://en.wikipedia.org'
    # Construct the matrices from wikipedia data without preprocessing
    print("Crawling the web...")
    get_Finnish_Cities()
    print("Starting to construct TF-IDF Matrix without preprocessing...")
    Tfidf = TFIDF(list(Finnish_Cities.values()))
    Tfidf = Tfidf.transform(list(Finnish_Cities.values()))
    Tfidf_matrix = Tfidf.toarray()
    print(f'{Tfidf_matrix}\nShape of the TF-IDF matrix (rows, cols): {Tfidf_matrix.shape}\n')
    print("Starting to construct CountVectorizer Matrix without preprocessing...")
    BOW = BOW_model(list(Finnish_Cities.values()))
    BOW_transformed = BOW.transform(list(Finnish_Cities.values()))
    BOW_matrix = BOW_transformed.toarray()
    print(f'{BOW_matrix}\nShape of the CountVectorizer matrix (rows, cols): {BOW_matrix.shape}\n')

    #  Construct the matrices from wikipedia data with preprocessing
    print("Starting to construct TF-IDF Matrix with preprocessing (stopwords, stemming, & lemmatization)...")
    Tfidf = TFIDF(list(Finnish_Cities.values()), preprocessor=preProcess)
    Tfidf = Tfidf.transform(list(Finnish_Cities.values()))
    Tfidf_matrix = Tfidf.toarray()
    print(f'{Tfidf_matrix}\nShape of the TF-IDF matrix (rows, cols): {Tfidf_matrix.shape}\n')
    print("Starting to construct CountVectorizer Matrix with preprocessing (stopwords, stemming, & lemmatization)...")
    BOW = BOW_model(list(Finnish_Cities.values()), preprocessor=preProcess)
    BOW_transformed = BOW.transform(list(Finnish_Cities.values()))
    BOW_matrix = BOW_transformed.toarray()
    print(f'{BOW_matrix}\nShape of the CountVectorizer matrix (rows, cols): {BOW_matrix.shape}\n')

    # Construct TF-IDF Matrix for a given query
    Tfidf = TFIDF(query)
    Tfidf = Tfidf.transform(query)
    Tfidf_matrix = Tfidf.toarray()
    print(f'TF-IDF Matrix for query {query}:')
    print(f'{Tfidf_matrix}\nShape of the TF-IDF Matrix (rows, cols): {Tfidf_matrix.shape}\n')
    # Evaluate the query to each wikipedia document
    print(f'Starting to evaluate query {query} to each Wikipedia document...')
    Tfidf = TFIDF(list(Finnish_Cities.values()), preprocessor=preProcess)
    print('_____________Using TF-IDF_____________')
    evaluate_query(query, Tfidf)
    BOW = BOW_model(list(Finnish_Cities.values()), preprocessor=preProcess)
    print('______________Using CountVectorizer______________')
    evaluate_query(query, BOW)
