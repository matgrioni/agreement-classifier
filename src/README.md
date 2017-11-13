# Introduction

This is a simple project that uses a subsection of the IAC (Internet Agreement Corpus) and a Naive Bayes algorithm to create a classifier that classifies responses that agree or disagree with a previous comment.

As stated, the current implementation uses Naive Bayes with a bag of words assumption. There are some slight modifications to note. Stems are used instead of words through the LancasterStemmer implementation on NLTK. Bigrams are also used as features to get a small context window. Finally note not all words are used, only a beginning substring of words and bigrams are considered.

This implementation has about 73.42% accuracy.


## TODOs

* Use entirety of IAC rather than smaller balanced set.
* Try to weasel out a little more performance out of the Naive Bayes Classifier.
* Use a new model to try to improve accuracy.


Enjoy!

-- Matias Grioni
