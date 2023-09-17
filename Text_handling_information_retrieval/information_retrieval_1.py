'''
Script with examples for information retrieval.

-Mikko Lempinen
'''
import nltk
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz

def query_matching(query, file):
    '''
    Check the given file for each keyword in the given query.
    :param query: (list) List of strings considered as keywords.
    :param file: (str) Name of the .txt file to search for give query.
    :return: (list) List with the same length as 'query', containing either 1 or 0 for each keyword. 
        1 = keyword was found in the file, 0 = keyword was not found in the file.
    '''
    result = []
    with open(file, 'r', encoding="utf8") as f:
        text = f.read()
    for i in query:
        if i in text:
            result.append(1)
        else:
            result.append(0)
    return result

def create_inverted_index(documents):
    '''
    Creates an inverted index of all of the given documents. Converts all tokens/words to lowercase.
    :param documents: (list) List of strings, where each string is a name of a text document without 
    the .txt ending.
    '''
    inverted_index = {}
    for doc in documents:
        with open(doc + '.txt', 'r', encoding="utf8") as file:
            contents = file.read()
        try:
            tokens = word_tokenize(contents)
        except LookupError:
            nltk.download('punkt')
            tokens = word_tokenize(contents)
        tokens = map(str.lower, tokens)
        for token in tokens:
            if token in inverted_index.keys():
                if doc not in inverted_index[token]:
                    inverted_index[token].append(doc)
            else:
                inverted_index[token] = [doc]
    return inverted_index

def get_keyword_docs(keyword, inverted_index):
    '''
    Get names of documents containing given keyword.
    :return: (list) List of strings.
    '''
    if keyword in inverted_index.keys():
        return inverted_index[keyword]
    return []

def get_keyword_docs_90(keyword, inverted_index):
    '''
    Get names of documents containing given keyword. Document is considered containing the keyword,
    if the Levenshtein distance between the two strings is > 90.
    :return: (list) List of strings.
    '''
    result = []
    for i in inverted_index.keys():
        fuzzratio = fuzz.ratio(keyword, i)
        if fuzzratio > 90:
            for j in inverted_index[i]:
                if j not in result:
                    result.append(j)
    return result

if __name__ == '__main__':
    # Logical query-matching
    print("\n\n------------------------------------- Test of logical query-matching -------------------------------------")
    query = ["web", "power", "elite", "network", "war"]
    file_name = "academic_journal_abstracts.txt"
    print(f'Given query: {query}\nGiven file name: {file_name}')
    query_match = query_matching(query, file_name)
    print(f'Logical query-matching vector: {query_match}')
    # Create the inverted index from abstract documents A0 - A19
    documents = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19']
    inverted_index = create_inverted_index(documents)
    # Get the keywords into a list
    with open('academic_journal_keywords.txt', 'r', encoding="utf8") as file:
        academic_keywords = file.read()
    academic_keywords = academic_keywords.replace(" ", "")
    academic_keywords = academic_keywords.split(",")
    # Output the list of files containing each keyword:
    print("\n\n------------------------------------- Lists of files containing each keyword -------------------------------------")
    for keyword in academic_keywords:
        docs = get_keyword_docs(keyword.lower(), inverted_index)
        print(f'Keyword "{keyword.lower()}" is found in the following abstract files: {docs}')
    # Output the list of files containing each keyword. Document is considered to contain the keyword, if the Levenshtein distance between the keyword and a token in a document is over 90.
    print("\n\n------------------------------------- Lists of files containing most of each keyword (Levenshtein distance over 90) -------------------------------------")
    for keyword in academic_keywords:
        docs = get_keyword_docs_90(keyword.lower(), inverted_index)
        print(f'Keyword "{keyword.lower()}" is found in the following abstract files: {docs}')
