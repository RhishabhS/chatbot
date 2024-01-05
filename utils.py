import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
# nltk.download('punkt')
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
def stem(word):
    stemmer=PorterStemmer()
    return stemmer.stem(word)
def bagOfWords(tsentence, vocab):
    #tsentence is tokenised sentence -> list of tokens
    #vocab -> dict -> {(stemmed) word: index}
    tsentence=list(map(stem, tsentence))
    bag=np.zeros(len(vocab))
    for word in tsentence:
        if(word in vocab.keys()):
            bag[vocab[word]]=1.0
    return bag
