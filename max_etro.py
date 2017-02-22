__author__ = 'LakshanD'

import csv, random
import nltk
import collections
import nltk.metrics
from nltk.tokenize import word_tokenize
word_features=[]

def main():
    numIterations = 100
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

    training_set_formatted = [(list_to_dict(element[0]), element[1]) for element in train_tweets]
    algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
    classifier = nltk.MaxentClassifier.train(training_set_formatted, algorithm, max_iter=numIterations)
    classifier.show_most_informative_features(10)

    for review in v_test:
        label = review[1]
        text = review[0]
        determined_label = classifier.classify(text)
        print determined_label, label


def list_to_dict(words_list):
    return dict([(word, True) for word in words_list])

'''
def create_confussion_matrix(v_test,classifier):
    test_truth   = [s for (t,s) in v_test]
    test_predict = [classifier.classify(extract_features(t.split())) for (t,s) in v_test]
    confussion_matrix=nltk.ConfusionMatrix(test_truth, test_predict )
    print 'Confusion Matrix'
    print(confussion_matrix)
    accuracy=(float)(confussion_matrix._correct)/confussion_matrix._total
    print("Accuracy:",accuracy)
'''



if __name__ == '__main__':
    main()
