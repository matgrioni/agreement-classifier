import nltk

class Stemmer:
    """
    A wrapper around a stemmer. This allows us to not worry about how to
    instantiate the stemmer and focus on its interface.
    """
    def __init__(self):
        self.internal = nltk.stem.LancasterStemmer()

    def stem(self, word):
        """
        Given a word, what is its stem.

        Args:
        word: The word to stem.

        Returns:
        The stem of the given word.
        """
        return self.internal.stem(word)
