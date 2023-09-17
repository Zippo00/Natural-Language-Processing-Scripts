'''
Script with examples for data processing and basic text handling.

-Mikko Lempinen
'''
import pandas as pd
import nltk
import pickle
import matplotlib.pyplot as plt
from nltk.corpus import brown
from nltk.corpus import stopwords

# Check that the brown corpus is downloaded
try:
    brown_categories = brown.categories()
except LookupError:
    nltk.download('brown')
    brown_categories = brown.categories()

# Extract individual word frequencies from the brown corpus
# brown_words = brown.words(categories=brown_categories)
# brown_freqs = nltk.FreqDist(brown_words)
brown_sents = brown.sents(categories=brown_categories)

# Save the frequencies as a pickle file
# with open('brown_frequencies.pkl', 'wb') as file:
#     pickle.dump(brown_freqs, file)

# Load the previously created pickle file containing frequencies of Brown corpus words.
with open('brown_frequencies.pkl', 'rb') as file:
    brown_freqs = pickle.load(file)

# Plot the 30 most common words and their frequencies
common_freqs_30 = brown_freqs.most_common(30)
df = pd.DataFrame(common_freqs_30, columns=['Word', 'Frequency'])
plt.figure(figsize=(15, 5))
plt.bar(df["Word"], df["Frequency"], align='edge', width=0.3)
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.title('Frequencies of the 30 most common words in Brown Corpus.')
plt.show()

# Plot the next 30, less common, words and their frequencies
common_freqs_60 = brown_freqs.most_common(60)
df = pd.DataFrame(common_freqs_60[30:], columns=['Word', 'Frequency'])
plt.figure(figsize=(15, 5))
plt.bar(df["Word"], df["Frequency"], align='edge', width=0.3)
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.title('Frequencies of the next 30, less common, words in Brown Corpus.')
plt.show()

# Plot 30 words and their frequencies from the midrange of word frequencies
# Sort all the words based on frequency
freqs = brown_freqs.items()
freqs = sorted(freqs, key = lambda x: x[1], reverse=True)
midd_freqs = []
# The corpus has 56057 different words --> midway point at 28028. We take words with index 28027 +-15
for i, j in enumerate(freqs):
    if i > 28012:
        midd_freqs.append(j)
        if i > 28041:
            break
# Plot the words & frequencies
df = pd.DataFrame(midd_freqs, columns=['Word', 'Frequency'])
plt.figure(figsize=(20, 5))
plt.bar(df["Word"], df["Frequency"], width=0.3)
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.title('Frequencies of the middle 30 words in Brown Corpus sorted by most frequent.')
plt.show()

# Sum up frequencies of words with same length
freqs_by_len = {}
for i in freqs:
    if len(i[0]) in freqs_by_len.keys():
        freqs_by_len[len(i[0])] += i[1]
    else:
        freqs_by_len[len(i[0])] = i[1]
# Plot the word frequencies by word length
freqs_by_len_list = freqs_by_len.items()
freqs_by_len_list = sorted(freqs_by_len_list, key = lambda x: x[1], reverse=True)
df = pd.DataFrame(freqs_by_len_list, columns=['Word Length', 'Frequency'])
#plt.figure(figsize=(20, 5))
plt.bar(df["Word Length"], df["Frequency"], width=0.3)
plt.xlabel("Word Length (No. of characters)")
plt.ylabel("Frequency")
plt.title('Frequencies of the words in Brown Corpus by word length.')
plt.show()

# Calculate the frequency of given modal words
modal_words = ['will', 'must', 'might', 'may', 'could', 'can']
modal_freqs = []
for i in freqs:
    if i[0] in modal_words:
        modal_freqs.append((i[0], i[1]))
print("\n\n----------------------------------Frequency of each modal word in the Brown corpus.----------------------------------\n")
print(modal_freqs)
modal_sentence_lengths = []
# If a modal word appears in a sentence, save the sentece length in terms of words AND characters into modal_sentence_lengths.
for sentence in brown_sents:
    for modal_word in modal_words:
        if modal_word in sentence:
            char_len = 0
            for i in sentence:
                char_len += len(i)
            modal_sentence_lengths.append((len(sentence), char_len))
            break
print("\n\n----------------------------------Lengths of sentences containing a modal word in terms of (words, characters).----------------------------------\n")
print(modal_sentence_lengths)

# Calculate number of stopwords in each sentence in the Brown corpus, and the length of said sentence in terms of words AND characters
brown_sentence_stats = []
try:
    stopwords = stopwords.words("english")
except LookupError:
    nltk.download('stopwords')
    stopwords = stopwords.words("english")
#print(len(stopwords))
print("Starting to process sentences in the Brown corpus. This may take a while...\n")
for sentence in brown_sents:
    stopword_count = 0
    char_len = 0
    for stopword in stopwords:
        if stopword in sentence:
            stopword_count += 1
    for i in sentence:
        char_len += len(i)
    brown_sentence_stats.append((len(sentence), char_len, stopword_count))

# Print pandas df containing Word length, Character length & Stopword count for each sentence
df = pd.DataFrame(brown_sentence_stats, columns=['Sentence Word Length', 'Sentence Character Length', 'Sentence Stopword Count'])
print(df)
