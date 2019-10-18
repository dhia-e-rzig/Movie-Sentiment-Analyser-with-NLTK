# Movie Sentiment Analyser
NLP simple project using NLTK

### Introduction: 
sentiment analysis model that can classify  movie reviews as either positive or negative. trained the model using 6000 IMDB movie reviews.

### Process: 

The Algorithm : 
Tokenize, clean and lemmatize the data and took only the adjectives from the reviews. 
Created a frequency distribution and found the most used words in all of the reviews. The top 6000 was used as the features.  

Make features vectors:
Created a dictionary representation of each review. The key in the dictoinary was each of the top words and the corresponding value was True of False for whether 'Word was in the review or not' 
Divdided the data into train test split (80/20). 
Use 5 different classification models to train on the data.  

Naive Bayes (NB)
Multinomial NB
Bernoulli NB
Logistic Regression
Stochastic Gradient Descent Classifier - SGD
Support Vector Classification - SVC


Trainer.py:  The script responsible for training and pickling the models used 
Live_classifier.py:  The script that loads the pickled models and  runs a small demo of the classfication process and is  responsible for classifying live from the console input. 


# Acknowledgement

 * Followed this amazing tutorial for the code: https://pythonprogramming.net/sentiment-analysis-module-nltk-tutorial/
 * Dataset - http://ai.stanford.edu/~amaas/data/sentiment/
 
   






