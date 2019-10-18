import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


# Defininig the ensemble model class

class EnsembleClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


documents_f = open("pickled_algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features4k_f = open("pickled_algos/word_features4k.pickle", "rb")
word_features = pickle.load(word_features4k_f)
word_features4k_f.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


# Load all classifiers from the pickled files

# function to load models given filepath
def load_model(file_path):
    classifier_f = open(file_path, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier


# Original Naive Bayes Classifier
ONB_Clf = load_model('pickled_algos/ONB_clf.pickle')

# Multinomial Naive Bayes Classifier
MNB_Clf = load_model('pickled_algos/MNB_clf.pickle')

# Bernoulli  Naive Bayes Classifier
BNB_Clf = load_model('pickled_algos/BNB_clf.pickle')

# Logistic Regression Classifier
LogReg_Clf = load_model('pickled_algos/LogReg_clf.pickle')

# Stochastic Gradient Descent Classifier
SGD_Clf = load_model('pickled_algos/SGD_clf.pickle')

ensemble_clf = EnsembleClassifier(ONB_Clf, MNB_Clf, BNB_Clf, LogReg_Clf, SGD_Clf)

# Initializing the ensemble classifier
ensemble_clf = EnsembleClassifier(ONB_Clf, MNB_Clf, BNB_Clf, LogReg_Clf, SGD_Clf)

# # List of only feature dictionary from the featureset list of tuples
# feature_list = [f[0] for f in testing_set]
#
# # Looping over each to classify each review
# ensemble_preds = [ensemble_clf.classify(features) for features in feature_list]
#
# f1_score(ground_truth, ensemble_preds, average = 'micro')


def sentiment(text):
    feats = find_features(text)
    return ensemble_clf.classify(feats), ensemble_clf.confidence(feats)



# sentiment analysis of reviews of captain marvel found on rotten tomatoes
text_a = '''The problem is with the corporate anticulture that controls these productions-and 
            the fandom-targeted demagogy that they're made to fulfill-which responsible casting 
                can't overcome alone.'''

text_b = '''"Everything was beautiful and nothing hurt"'''
print(" Here are some example reviews of captain marvel : ")
print(text_a)
print(sentiment(text_a))
print(text_b)
print(sentiment(text_b))
print("what did you think ?")
text_c = input()
print(sentiment(text_c))




