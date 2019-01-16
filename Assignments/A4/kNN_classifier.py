

# Import the PySpark package

from pyspark import SparkContext
sc = SparkContext()

# The provided code to create the dictionary

import re
import numpy as np

# load up all of the  26,754 documents in the corpus
# corpus = sc.textFile ("s3://risamyersbucket/A4/pubmed.txt")
corpus = sc.textFile('./Data/pubmed.txt')

# each entry in validLines will be a line from the text file
validLines = corpus.filter(lambda x : 'id=' in x)

# now we transform it into a bunch of (docID, text) pairs
keyAndText = validLines.map(lambda x : (x[x.index('id=') + 3 : x.index('> ')], x[x.index('> ') + 2:]))

# now we split the text in each (docID, text) pair into a list of words
# after this, we have a data set with (docID, ["word1", "word2", "word3", ...])
# we have a bit of fancy regular expression stuff here to make sure that we do not
# die on some of the documents
regex = re.compile('[^a-zA-Z]')
keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

# now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
# to ("word1", 1) ("word2", 1)...
allWords = keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))

# now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
allCounts = allWords.reduceByKey (lambda a, b: a + b)

# and get the top 20,000 words in a local array
# each entry is a ("word1", count) pair
topWords = allCounts.top (20000, lambda x : x[1])

# and we'll create a RDD that has a bunch of (word, dictNum) pairs
# start by creating an RDD that has the number 0 thru 20000
# 20000 is the number of words that will be in our dictionary
twentyK = sc.parallelize(range(20000))

# now, we transform (0), (1), (2), ... to ("mostcommonword", 1) ("nextmostcommon", 2), ...
# the number will be the spot in the dictionary used to tell us where the word is located
# HINT: make use of topWords in the lambda that you supply
dictionary = twentyK.map(lambda x:(topWords[x][0],x))


# finally, print out some of the dictionary, just for debugging
# dictionary.top (10)

#

# TASK ONE

#

# Function to convert final_dict into a numpy array

def rdd_to_np_array(dict_doc_rdd):

    res = np.zeros(20000)

    for i in dict_doc_rdd:

        res[i] += 1

    return res


# Create pairs of words and documents from the most common words

pairs_word_doc = keyAndListOfWords.flatMap(lambda doc: ((word, doc[0]) for word in doc[1]))

# Join the dictionaries together by word

word_dict = dictionary.join(pairs_word_doc)

# Map the dictionary into pairs of doc and word counts for each word

doc_count_dict = word_dict.map(lambda x: (x[1][1], x[1][0]))

# Group by key to have the number of the 20000 words per document

final_dict = doc_count_dict.groupByKey()

# Perform the conversion to output the results

final_counts = final_dict.map(lambda dic: (dic[0], rdd_to_np_array(dic[1])))
result_1 = final_counts.lookup('Wounds/23778438')
result_2 = final_counts.lookup('ParasiticDisease/2617030')
result_3 = final_counts.lookup('RxInteractions/1966748')

# Output the results

print()
print('Task 1 Results:')
print()
print('Wounds/23778438:')
print()
print(result_1[0][result_1[0].nonzero()])
print()
print('ParasiticDisease/2617030:')
print()
print(result_2[0][result_2[0].nonzero()])
print()
print('RxInteractions/1966748:')
print()
print(result_3[0][result_3[0].nonzero()])

#

# TASK TWO

#

# Calculate the term frequency

term_freq = final_counts.map(lambda dic: (dic[0], np.divide(dic[1], np.sum(dic[1]))))

# Construct a multi-dimensional array to hold the results

final_mat = np.full(20000, 26754.0)

# Return either 0 or 1 depending on the dict value

dict_indict = final_counts.map(lambda dic: (dic[0], np.where(dic[1] > 0, 1, 0)))

# Reduce the elements of the RDD down and retrieve the values

reduced = dict_indict.reduce(lambda a, b: ('', np.add(a[1], b[1])))[1]

# Calculate the inverse document frequency

inv_doc_freq = np.log(np.divide(final_mat, reduced))

# Calculate the term frequency-inverse document frequency

tf_idf = term_freq.map(lambda val: (val[0], np.multiply(val[1], inv_doc_freq)))
result_1 = tf_idf.lookup('PalliativeCare/16552238')
result_2 = tf_idf.lookup('SquamousCellCancer/23991972')
result_3 = tf_idf.lookup('HeartFailure/25940075')

# Output the results

print()
print('Task 2 Results:')
print()
print('PalliativeCare/16552238:')
print()
print(result_1[0][result_1[0].nonzero()])
print()
print('SquamousCellCancer/23991972:')
print()
print(result_2[0][result_2[0].nonzero()])
print()
print('HeartFailure/25940075:')
print()
print(result_3[0][result_3[0].nonzero()])

#

# TASK THREE

#


# Define a function to calculate the euclidean distance that we can pass as a lambda

def euclidean_dist(x, y):

    return np.linalg.norm(x - y)


# Define a function to tally the total word counts that we can pass as a lambda

def total_count(line):

    total_words = {}

    for word in line:

        total_words[word] = total_words.get(word, 0) + 1

    words_dict = np.zeros(20000)

    for word in total_words.keys():

        if word in words_final:

            words_dict[words_final.index(word)] = total_words[word]

    return words_dict


# Load in the queries files

queries_file = sc.textFile('./Data/A4Queries.txt')

# Load the queries in as a dictionary

lines = queries_file.filter(lambda line: '10, ' in line)

doc_text_dict = lines.map(lambda line: (line[line.index('10, ') + 5:]))

queries_file = doc_text_dict.map(lambda line: (regex.sub(' ', line).lower().split()))

# Get a list of the dictionary keys as words

words_final = dictionary.keys().collect()

# Apply the total_count() function to the extracted queries

queries_rdd = queries_file.map(lambda word: (total_count(word)))

# Calculate the term frequency-inverse document frequency

tf = queries_rdd.map(lambda line: line / np.sum(line))
tf_idf_final = tf.map(lambda line: np.multiply(line, inv_doc_freq)).collect()

# Container for the results of the kNN analysis

results = list()

# Perform the kNN analysis by looping through the kNN labels results and finding the top label

for elem in range(len(tf_idf_final)):

    kNN_result = tf_idf.map(lambda line: (euclidean_dist(line[1], tf_idf_final[elem]), line[0]))
    kNN_final = kNN_result.sortByKey().take(10)

    kNN_labels = list()

    for drop, elem in kNN_final:

        label = elem.split('/')
        kNN_labels.append(label[0])

    labels = {}
    top_lab = ('', 0)

    for label in kNN_labels:

        # Add 1 if our label is already present or set to 1 if not

        if label in labels:

            labels[label] += 1

        else:

            labels[label] = 1

        # If our label is the top result, set to our current top container

        if labels[label] > top_lab[1]:

            top_lab = (label, labels[label])

    results.append(top_lab[0])

# Output the results

print()
print('Task 3 Results:')
print()

for i in results:

    print(i)
