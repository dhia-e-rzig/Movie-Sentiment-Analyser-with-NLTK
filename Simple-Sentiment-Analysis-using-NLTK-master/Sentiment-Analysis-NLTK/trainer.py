import os
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
import re

# importing reviews
print("Loading Files...")
files_pos = os.listdir('train/pos')
files_pos = [open('train/pos/' + f, 'r', encoding="utf8").read() for f in files_pos]
files_neg = os.listdir('train/neg')
files_neg = [open('train/neg/' + f, 'r', encoding="utf8").read() for f in files_neg]
print("Files Loaded")
all_words = []
documents = []

from nltk.corpus import stopwords
import re

# importing list of stopwords ( most common words in a language,E.G : this,that,at...)
stop_words = list(set(stopwords.words('english')))

#  j is adject, r is adverb, and v is verb
# allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]
# adjs are used as a sentiment indicator

print("Loading Adjectives...")
for p in files_pos:

    # create a list of tuples (review,label)
    documents.append((p, "pos"))

    # remove punctuations
    cleaned = re.sub(r'[^(a-zA-Z)\s]', '', p)

    # tokenize : split words in different objects instead of continuous strings
    tokenized = word_tokenize(cleaned)

    # remove stopwords
    stopped = [w for w in tokenized if not w in stop_words]

    # parts of speech tagging for each word (identifying whether word is noun,verb or adjective)
    pos = nltk.pos_tag(stopped)
    # make a list of  all adjectives identified by the allowed word types list above
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

# same process as above for negative reviews
for p in files_neg:
    documents.append((p, "neg"))
    cleaned = re.sub(r'[^(a-zA-Z)\s]', '', p)
    tokenized = word_tokenize(cleaned)
    stopped = [w for w in tokenized if not w in stop_words]
    neg = nltk.pos_tag(stopped)
    # make a list of  all adjectives identified by the allowed word types list above
    for w in neg:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
print("Adjectives Loaded")

pos_A = []
# make a list of  all positive adjectives identified by the allowed word types list above
for w in pos:
    if w[1][0] in allowed_word_types:
        pos_A.append(w[0].lower())

pos_N = []
# make a list of  all negative adjectives identified by the allowed word types list above
for w in neg:
    if w[1][0] in allowed_word_types:
        pos_N.append(w[0].lower())

# pickling(serializing) the list documents to save future recalculations

save_documents = open("pickled_algos/documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

# creating a frequency distribution of each adjectives.
BOW = nltk.FreqDist(all_words)

# saving the 4000 most frequent words ( more words will usually give better results)
word_features = list(BOW.keys())[:4000]
save_word_features = open("pickled_algos/word_features4k.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

# function to create a dictionary of features for each review in the list document.
# The keys are the words in word_features ( the 4000 most frequent words in our reviews)
# The values of each key are either true or false for whether that feature appears in the review or not

print("Finding  Features...")


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


print("Features Found")

print("Assigning  Features...")
# Creating features for each review
featuresets = [(find_features(rev), category) for (rev, category) in documents]
print("Features Assigned ")

print("Shuffling Documents...")
# Shuffling the documents
random.shuffle(featuresets)
print("Documents Shuffled")
# dividing into training / testing set with the 80/20 ratio

training_set = featuresets[:20000]
testing_set = featuresets[20000:]
print('training_set :', len(training_set), '\ntesting_set :', len(testing_set))
print("Running  NB...")
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("NB  Done")
print("Naive Bayes Classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)

# Printing the most important features

# mif = classifier.most_informative_features()
#
# mif = [a for a,b in mif]
# print(mif)

# calculating f1 score for the NB classifier

ground_truth = [r[1] for r in testing_set]
preds = [classifier.classify(r[0]) for r in testing_set]
from sklearn.metrics import f1_score
# f1_score(ground_truth, preds, labels = ['neg', 'pos'], average = 'micro')

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC

# Classifiers for an ensemble model:
# Naive Bayes (NB)
# Multinomial NB
# Bernoulli NB
# Logistic Regression
# Stochastic Gradient Descent Classifier - SGD
# Support Vector Classification - SVC


# training the different classifiers
print("Running  MNB...")
MNB_clf = SklearnClassifier(MultinomialNB())
MNB_clf.train(training_set)
print("MNB Done")
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_clf, testing_set)) * 100)

print("Running BernoulliNB...")
BNB_clf = SklearnClassifier(BernoulliNB())
BNB_clf.train(training_set)
print(" BernoulliNB Done")
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BNB_clf, testing_set)) * 100)

print("Running LogisticRegression...")
LogReg_clf = SklearnClassifier(LogisticRegression())
LogReg_clf.train(training_set)
print(" LogisticRegression Done")
print("LogisticRegressionaccuracy percent:", (nltk.classify.accuracy(LogReg_clf, testing_set)) * 100)

print("Running SGDClassifier...")
SGD_clf = SklearnClassifier(SGDClassifier())
SGD_clf.train(training_set)
print(" SGDClassifier Done")
print("SGDClassifier accuracy percent:", (nltk.classify.accuracy(SGD_clf, testing_set)) * 100)

print("Running SVC_classifier...")
SVC_clf = SklearnClassifier(SVC())
SVC_clf.train(training_set)
print(" SVC_classifier Done")
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_clf, testing_set)) * 100)


### Storing all models using pickle


def create_pickle(c, file_name):
    save_classifier = open(file_name, 'wb')
    pickle.dump(c, save_classifier)
    save_classifier.close()


classifiers_dict = {'ONB': [classifier, 'pickled_algos/ONB_clf.pickle'],
                    'MNB': [MNB_clf, 'pickled_algos/MNB_clf.pickle'],
                    'BNB': [BNB_clf, 'pickled_algos/BNB_clf.pickle'],
                    'LogReg': [LogReg_clf, 'pickled_algos/LogReg_clf.pickle'],
                    'SGD': [SGD_clf, 'pickled_algos/SGD_clf.pickle'],
                    'SVC': [SVC_clf, 'pickled_algos/SVC_clf.pickle']}

for clf, listy in classifiers_dict.items():
    create_pickle(listy[0], listy[1])

# calculating f accuracy and f1 scores for each of the classifiers
from sklearn.metrics import f1_score, accuracy_score

predictions = {}
acc_scores = {}


for clf, listy in classifiers_dict.items():
    # getting predictions for the testing set by looping over each reviews featureset tuple
    # The first element of the tuple is the feature set and the second element is the label
    acc_scores[clf] = accuracy_score(ground_truth, predictions[clf])
    print(f'Accuracy_score {clf}: {acc_scores[clf]}')

ground_truth = [r[1] for r in testing_set]

f1_scores = {}
for clf, listy in classifiers_dict.items():
    # getting predictions for the testing set by looping over each reviews featureset tuple
    # The first elemnt of the tuple is the feature set and the second element is the label
    predictions[clf] = [listy[0].classify(r[0]) for r in testing_set]
    f1_scores[clf] = f1_score(ground_truth, predictions[clf], labels=['neg', 'pos'], average='micro')
    print(f'f1_score {clf}: {f1_scores[clf]}')
