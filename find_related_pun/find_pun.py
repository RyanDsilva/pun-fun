import numpy as np
from typing import List
import math
import pandas as pd
from pathlib import Path
import os

from classes import *

PROJECT_ROOT = Path().parent.absolute()
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, 'embeddings')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

GLOVE = 'glove.6B.300d.txt'
FREQUENCY_LIST = 'word-frequency-list.txt'
SEMEVAL_TASK3_HOMO = 'semeval-task3-homo.json'
SEMEVAL_TASK3_HETERO = 'semeval-task3-hetero.json'

EMBEDDING_SIZE = 300

def read_data():
    d1 = pd.read_json(os.path.join(DATA_DIR, SEMEVAL_TASK3_HOMO), orient='records')
    d2 = pd.read_json(os.path.join(DATA_DIR, SEMEVAL_TASK3_HETERO), orient='records')

    d1 = d1[['sentence']]
    d1 = d1.replace(' \- ', '-', regex=True)
    d1 = d1.replace(' \' ', "'", regex=True)
    d1 = d1.replace(' \. ', '. ', regex=True)
    d1 = d1.replace(' \? ', '? ', regex=True)
    d1 = d1.replace(' \! ', '! ', regex=True)
    d1 = d1.replace(' \, ', ', ', regex=True)

    d2 = d2[['sentence']]
    d2 = d2.replace(' \- ', '-', regex=True)
    d2 = d2.replace(' \' ', "'", regex=True)
    d2 = d2.replace(' \. ', '. ', regex=True)
    d2 = d2.replace(' \? ', '? ', regex=True)
    d2 = d2.replace(' \! ', '! ', regex=True)
    d2 = d2.replace(' \, ', ', ', regex=True)

    df = pd.concat([d1, d2])
    data = df['sentence'].tolist()
    return data

def build_word_frequency_and_average():
    word_frequency = dict()
    avg_frequency = 1.0
    with open(os.path.join(EMBEDDINGS_DIR, FREQUENCY_LIST), "rt") as reader:
        # max_value = 0.0
        counter = 0
        for line in reader:
            line = line.strip().split(" ")
            if len(line) == 2:
                value = math.log2(float(line[1]))
                avg_frequency += value
                counter += 1
                word_frequency[line[0].lower()] = value
        avg_frequency /= counter
    return word_frequency, avg_frequency


def get_word_frequency(word_text, word_frequency, avg_frequency):
    if word_text.lower() in word_frequency:
        return word_frequency[word_text.lower()]
    else:
        return avg_frequency


# convert a list of sentence with glove vectors into a set of sentence vectors
def sentence_to_vec(sentence_list: List[Sentence], embedding_size: int):
    if len(sentence_list) == 0:
        return []
    sentence_set = []
    delta = 0.001  # small value to avoid division by 0
    for sentence in sentence_list:
        vs = np.zeros(
            embedding_size
        )  # add all glove values into one vector for the sentence
        sentence_length = 0.0
        for word in sentence.word_list:
            # basically the importance of a word becomes less the more frequent it is
            a_value = delta / (
                delta + get_word_frequency(word.text)
            )  # smooth inverse frequency, SIF
            sentence_length += a_value
            vs = np.add(
                vs, np.multiply(a_value, word.vector)
            )  # vs += sif * word_vector
        if sentence_length != 0.0:
            vs = np.divide(vs, sentence_length)  # weighted average
        sentence_set.append(vs)  # add to our existing re-calculated set of sentences
    return sentence_set


def inner_product(v1, v2):
    if len(v1) == len(v2):
        sum = 0.0
        size_v1 = 0.0
        size_v2 = 0.0
        for i in range(len(v1)):
            size_v1 += v1[i] * v1[i]
            size_v2 += v2[i] * v2[i]
            sum += v1[i] * v2[i]
        size_v1 = math.sqrt(size_v1)
        size_v2 = math.sqrt(size_v2)
        size_mult = size_v1 * size_v2
        if size_mult != 0.0:
            return round(sum / size_mult, 4)
    return 0.0


def load_glove():
    # load the glove set from file
    glove_300_dict = dict()
    with open(os.path.join(EMBEDDINGS_DIR, GLOVE), 'rt') as reader:
        for line in reader:
            line = line.strip().split(' ')
            if len(line) == (EMBEDDING_SIZE + 1):
                word = line[0]
                vector = [float(item) for item in line[1:]]
                glove_300_dict[word] = vector
    return glove_300_dict

def build_sentence_embeddings(sentences, glove_dict):
    # convert the above sentences to vectors using spacy's large model vectors
    sentence_list = []
    for sentence in sentences:
        word_list = []
        for word in sentence.split(' '):
            if word.lower() in glove_dict:  # ignore OOVs
                word_list.append(Word(word, glove_dict[word.lower()]))
        if len(word_list) > 0:  # did we find any words (not an empty set)
            sentence_list.append(Sentence(word_list))
    # apply single sentence word embedding
    sentence_vector_lookup = dict()
    sentence_vectors = sentence_to_vec(sentence_list, EMBEDDING_SIZE)  # all vectors converted together
    if len(sentence_vectors) == len(sentence_list):
        for i in range(len(sentence_vectors)):
            # map: text of the sentence -> vector
            sentence_vector_lookup[sentence_list[i]] = sentence_vectors[i]
    return sentence_vector_lookup