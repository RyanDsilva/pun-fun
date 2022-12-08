import nltk
import re
from string import digits
from pathlib import Path
import os
import numpy as np
from scipy.spatial import distance

PROJECT_ROOT = Path().parent.absolute()
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, 'embeddings')

V_EMB = 'similarityV.txt'
C_EMB = 'similarityC.txt'

def read_embedding(path, weight = 1):
    embedding = {}
    raw_file = open(os.path.join(EMBEDDINGS_DIR, path), 'rt')
    for line in raw_file:
        l = line.strip().split(',')
        key = l[0]
        val = [int(x) for x in l[1:]]
        embedding[key] = np.array(val) * weight
    return embedding

arpabet = nltk.corpus.cmudict.dict()
remove_digits = str.maketrans('', '', digits)

VOWEL_EMBEDDING = read_embedding(V_EMB, 2)
CONSONANT_EMBEDDING = read_embedding(C_EMB)

def get_vector_representation(word):
    result = np.zeros((4,), dtype=int)
    for w in word:
        if w in VOWEL_EMBEDDING:
            result += VOWEL_EMBEDDING[w]
        elif w in CONSONANT_EMBEDDING:
            result += CONSONANT_EMBEDDING[w]
    return result

def get_sound_similarity(source, target):
    src_arp = None
    tgt_arp = None
    try:
        src_arp = arpabet[source.lower()][0]
        tgt_arp = arpabet[target.lower()][0]
    except:
        return None
    if src_arp is not None and tgt_arp is not None:
        src_stress = sum([int(x) for x in re.findall('[0-9]+', ''.join(src_arp))])
        tgt_stress = sum([int(x) for x in re.findall('[0-9]+', ''.join(tgt_arp))])
        src_arp = [s.translate(remove_digits) for s in src_arp]
        tgt_arp = [s.translate(remove_digits) for s in tgt_arp]

        src_vec = get_vector_representation(src_arp)
        tgt_vec = get_vector_representation(tgt_arp)
        # print(src_vec)
        # print(tgt_vec)
        dist = distance.euclidean(src_vec, tgt_vec)
        # dist = distance.euclidean(src_vec, tgt_vec) / 120
        return dist, src_stress, tgt_stress

# print(get_sound_similarity('arm', 'psalm'))
