import collections
import math

import counter

class NaiveBayesClassifier:
    """
    A classifier that uses Naive Bayes calculations. The general structure of
    this classifier is to provide functions to featurize and to class sample
    data. This processed sample data is used to amass probability functions that
    will be used to classify new documents.
    """

    def __init__(self, featurizer, classer, classes):
        """
        Create a new classifier.

        Args:
        featurizer: A function argument that is given a data sample and returns
                    a list of features, which cannot ever be empty.
        classer: A function that returns the class of a data sample.
        classes: The possible classes the classifier will see. Since we might
                 not see all classes in the data, this needs to be explicitly
                 specified.
        """
        self.featurizer = featurizer
        self.classer = classer
        self.classes = classes

        self.class_counts = counter.Counter()
        self.feature_counts = collections.defaultdict(lambda: counter.Counter())
        self.class_to_feature_counts = counter.Counter()

    def add_sample(self, sample):
        """
        Add a new data sample.

        Args:
        sample: The data sample to take into account to the probability
                distribution.
        """
        cls = self.classer(sample)
        features = self.featurizer(sample)

        self.class_counts[cls] += 1
        for feature in features:
            self.feature_counts[feature][cls] += 1
            self.class_to_feature_counts[cls] += 1


    def smooth(self):
        """
        This smooths the data so that there are no zero counts. This uses basic
        laplace smoothing, where each count is incremented by one and the total
        size is incremented by the number of features we have seen. Call this
        after all data has been added to the classifier.
        """
        for feature, counter in self.feature_counts.items():
            for cls in self.classes:
                self.feature_counts[feature][cls] += 1

        for cls in self.class_to_feature_counts:
            self.class_to_feature_counts[cls] += len(self.feature_counts) + 1


    def classify(self, sample):
        """
        Classify the given sample among the possible classes.

        Args:
        sample: The sample to classify.

        Returns:
        The most likely class given the data seen so far.
        """
        features = self.featurizer(sample)

        argmax = float('-inf')
        maxcls = None
        for cls in self.classes:
            prior = self.class_counts.probability(cls)

            likelihood = 1
            for feat in features:
                count = self.feature_counts[feat][cls]
                if count == 0:
                    count = 1

                cond = count / self.class_to_feature_counts[cls]
                likelihood *= cond

            final = likelihood * prior

            if final > argmax:
                argmax = final
                maxcls = cls

        return maxcls
