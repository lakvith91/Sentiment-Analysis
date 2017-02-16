__author__ = 'LakshanD'

import csv, random
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

'''
Training a model using naive bayes and classify data
'''
word_features=[]

def main():

    # read all tweets and labels
    fp = open('apple.csv', 'rb')
    reader = csv.reader(fp, delimiter=',')
    raw_tweets = []
    train_tweets=[]
    for row in reader:
        raw_tweets.append((row[0], row[1]))

    random.shuffle(raw_tweets)
    v_train=raw_tweets[:len(raw_tweets)*75/100]
    v_test=raw_tweets[len(raw_tweets)*75/100:]


    #tweet preprocessed
    for (tweet, sentiment) in v_train:
        words_filtered=[]
        for word in word_tokenize(tweet):
            words_filtered.append(word)
        train_tweets.append((words_filtered,sentiment))

    global word_features
    #create feature vector using bag of word technique
    word_features=get_word_features(get_words(train_tweets))

    training_set=nltk.classify.apply_features(extract_features,train_tweets,labeled=True)
    classifier=nltk.NaiveBayesClassifier.train(training_set)
    #print '\nAccuracy %f\n' % nltk.classify.accuracy(classifier,v_test)
    #print(nltk.classify.accuracy(classifier, v_test))
    print classifier.show_most_informative_features(10)

    #Classify Test tweets using trained classifier
    #tee="loving new IOS 5 upgrade iPhone"
    #print(classifier.classify(extract_features(tee.split())))

    #plot confusion matrix and accuracy of the model
    create_confussion_matrix(v_test,classifier)


def get_words(tweet_set):
    all_words = []
    for (tweet, sentiment) in tweet_set:
        all_words.extend(tweet)
    return all_words

def get_word_features(all_words):
    word_freq=FreqDist(all_words)
    word_features=word_freq.keys()
    return word_features

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

def create_confussion_matrix(v_test,classifier):
    test_truth   = [s for (t,s) in v_test]
    test_predict = [classifier.classify(extract_features(t.split())) for (t,s) in v_test]
    confussion_matrix=nltk.ConfusionMatrix(test_truth, test_predict )
    print 'Confusion Matrix'
    print(confussion_matrix)
    accuracy=(float)(confussion_matrix._correct)/confussion_matrix._total
    print("Accuracy:",accuracy)


if __name__ == '__main__':
    main()
