import nltk

class Stemmer:
    def __init__(self):
        self.internal = nltk.stem.LancasterStemmer()

    def stem(self, word):
        return self.internal.stem(word)
