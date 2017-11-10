import collections
import math

import counter

class NaiveBayesClassifier:
    def __init__(self, featurizer=lambda x: [x], classer=lambda x: x):
        self.featurizer = featurizer
        self.classer = classer

        self.class_counts = counter.Counter()
        self.feature_counts = collections.defaultdict(lambda: counter.Counter())

    def add_sample(self, sample):
        cls = self.classer(sample)
        features = self.featurizer(sample)

        self.class_counts[cls] += 1
        for feature in features:
            self.feature_counts[feature][cls] += 1


    def classify(self, sample):
        # TODO: What if len(features) == 0
        features = self.featurizer(sample)

        argmax = -math.inf
        maxcls = None
        for cls in self.class_counts.keys():
            prior = self.class_counts.probability(cls)

            likelihood = 1
            for feat in features:
                cond = self.feature_counts[feat][cls] / self.class_counts[cls]
                likelihood *= cond

            final = likelihood * prior

            if final > argmax:
                argmax = final
                maxcls = cls

        return maxcls
