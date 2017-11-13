#!/usr/bin/env python3

import argparse
import csv
import math
import re

from counter import Counter
from naivebayesclassifier import NaiveBayesClassifier
from stemmer import Stemmer


AGREE_CLASS = 'AGREE'
DISAGREE_CLASS = 'DISAGREE'

stemmer = Stemmer()


def classer(sample):
    """
    Returns the class of a given sample. This is to be used in the Naive Bayes
    classifier. A sample in this case is an item from the IAC data, which has
    been parsed as a csv row.

    Args:
    sample: The sample or data item in IAC. Consists of agreement, quote, and
            response.

    Returns:
    Provides the class of the given sample.
    """
    score = float(sample[1])
    if score >= 1 and score <= 5:
        return AGREE_CLASS
    elif score < -1 and score >= -5:
        return DISAGREE_CLASS


def featurizer(sample):
    """
    Feature the given sample item from the IAC data. In this case, the features
    I am using are the first 3 words of the response.

    Args:
    sample: The sample or data item in IAC. Consists of agreement, quote, and
            response.

    Returns:
    Provides the features of the given sample.
    """
    # Remove punctuation, convert into lowercase, and other miscellaneous
    # preprocessing.
    response = sample[3]
    processed = re.sub(r'&#8217;', r"'", response)
    processed = re.sub(r'&#8212;', r"-", response)
    processed = re.sub(r'([^\w\s\'])', r' \1 ', response)
    processed = processed.lower()
    words = processed.split()

    stems = list(map(lambda word: stemmer.stem(word), words))
    bistems = []
    for i, stem in enumerate(stems[:40]):
        if i + 1 < len(stems):
            next_stem = stems[i + 1]
        else:
            next_stem = None
        bistems.append((stem, next_stem))

    features = stems[:40]
    features.extend(bistems)

    return features


parser = argparse.ArgumentParser()
parser.add_argument('train', help='The filename that points to training set.')
parser.add_argument('test', help='The filename that points to test set.')
args = parser.parse_args()


# Train our classifier
nbc = NaiveBayesClassifier(featurizer, classer, (AGREE_CLASS, DISAGREE_CLASS))
with open(args.train, 'r') as csv_train:
    train_reader = csv.reader(csv_train, delimiter=',')
    next(train_reader)

    for row in train_reader:
        rating = float(row[1])
        if rating >= -1 and rating < 1:
            continue

        nbc.add_sample(row)
nbc.smooth()


false_counts = Counter()
true_counts = Counter()
real_counts = Counter()

# Now evaluate the trainied classifier.
with open(args.test, 'r') as csv_test:
    test_reader = csv.reader(csv_test, delimiter=',')
    next(test_reader)

    for row in test_reader:
        rating = float(row[1])
        if rating >= -1 and rating < 1:
            continue

        cls = nbc.classify(row)
        actual_cls = classer(row)

        real_counts[actual_cls] += 1

        if cls == actual_cls:
            true_counts[cls] += 1
        else:
            false_counts[cls] += 1

correct = 0
for cls, count in true_counts.items():
    correct += count

incorrect = 0
for cls, count in false_counts.items():
    incorrect += count

print('Accuracy: {}'.format(correct / (correct + incorrect)))
for cls in nbc.classes:
    print('Precision for {}: {}'.format(cls, true_counts[cls] / (true_counts[cls] + false_counts[cls])))
    print('Recall for {}: {}'.format(cls, true_counts[cls] / real_counts[cls]))
