import collections
import math

import counter

class NaiveBayesClassifier:
    def __init__(self, featurizer=lambda x: [x], classer=lambda x: x):
        self.featurizer = featurizer
        self.classer = classer

        self.class_counts = counter.Counter()
        self.feature_counts = collections.defaultdict(lambda: counter.Counter())
        self.class_to_feature_counts = counter.Counter()

    def add_sample(self, sample):
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
        # TODO: Make classes it's own thing so that it is not dependent on the
        # data we have seen so far.
        classes = self.class_counts.keys()

        for feature, counter in self.feature_counts.items():
            for cls in classes:
                self.feature_counts[feature][cls] += 1

        for cls in self.class_to_feature_counts:
            self.class_to_feature_counts[cls] += len(self.feature_counts) + 1


    def classify(self, sample):
        # TODO: What if len(features) == 0
        features = self.featurizer(sample)

        argmax = -math.inf
        maxcls = None
        for cls in self.class_counts.keys():
            prior = self.class_counts.probability(cls)

            likelihood = 1
            for feat in features:
                count = self.feature_counts[feat][cls]
                if count == 0:
                    count = 1

                cond = count / self.class_to_feature_counts[cls]
                likelihood *= cond

            final = likelihood * prior
            print('{}\t{}'.format(final, cls))

            if final > argmax:
                argmax = final
                maxcls = cls

        return maxcls
