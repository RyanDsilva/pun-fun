from nltk.corpus import PlaintextCorpusReader
import spacy
import re
import markovify
import nltk
from nltk.corpus import gutenberg
import warnings

corpus_root = 'shortjokes.csv'

filelists = PlaintextCorpusReader('.', '.*')

filelists.fileids()

sample = open("shortjokes.csv")
s = sample.read()

f = s.replace("\"", "")

generator_1 = markovify.Text(s, state_size=2)

for i in range(3):
    if i == 0:
        print(generator_1.make_sentence())
    else:
        print(generator_1.make_sentence('God is cruel'))
