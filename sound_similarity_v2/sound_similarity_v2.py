import nltk
import re
from string import digits
from pathlib import Path
import os
import numpy as np
from scipy.spatial import distance
from sklearn.decomposition import PCA
import pandas as pd

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
PCA_FUNC = PCA(n_components=3)

def PCA_and_convert(emb):
    x = pd.DataFrame(emb.keys())
    y = pd.DataFrame(PCA_FUNC.fit_transform(pd.DataFrame(emb.values())))
    p = pd.merge(x, y, left_index=True, right_index=True, how='inner')
    return p.set_index('0_x').T.to_dict('list')

VOWEL_EMBEDDING = PCA_and_convert(read_embedding(V_EMB, 2))
CONSONANT_EMBEDDING = PCA_and_convert(read_embedding(C_EMB))


def get_vector_representation(word):
    result = np.zeros((3,), dtype=float)
    for w in word:
        if w in VOWEL_EMBEDDING:
            # print(VOWEL_EMBEDDING[w])
            result += VOWEL_EMBEDDING[w]
        elif w in CONSONANT_EMBEDDING:
            # print(CONSONANT_EMBEDDING[w])
            result += CONSONANT_EMBEDDING[w]
    # print("===")
    return result

def get_vector_representation_crossed(word):
    # print(word[0], 'test')
    result = None
    if word[0] in VOWEL_EMBEDDING:
        result = VOWEL_EMBEDDING[word[0]]
    else:
        result = CONSONANT_EMBEDDING[word[0]]
    for i in range(1, len(word)):
        if word[i] in VOWEL_EMBEDDING:
            # print('v', VOWEL_EMBEDDING[word[i]])
            result = list(np.cross(result, VOWEL_EMBEDDING[word[i]]))
        elif word[i] in CONSONANT_EMBEDDING:
            # print('c', CONSONANT_EMBEDDING[word[i]])
            result = list(np.cross(result, CONSONANT_EMBEDDING[word[i]]))
    # print(result)
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
        # print(src_arp, tgt_arp)
        src_vec = get_vector_representation(src_arp)
        tgt_vec = get_vector_representation(tgt_arp)
        # print('enter')
        # print(src_vec)
        # print(tgt_vec)
        # r = np.cross(src_vec, tgt_vec)
        dist = distance.euclidean(src_vec, tgt_vec) / 120
        # dist = distance.cosine(src_vec, tgt_vec)
        return (dist, src_stress, tgt_stress)

# print(get_sound_similarity('bear', 'bare'))
# print(get_sound_similarity('axe', 'chocolate'))
# print(get_sound_similarity('bear', 'door'))
