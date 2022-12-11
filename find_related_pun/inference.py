import pickle
from pathlib import Path
import os
from classes import *
from find_pun import *

PROJECT_ROOT = Path().parent.absolute()
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, 'embeddings')

SENT_EMBEDDING = 'semeval-sent-vectors-hetero.pkl'

pkl_file = open(os.path.join(EMBEDDINGS_DIR, SENT_EMBEDDING), 'rb')
puns = pickle.load(pkl_file)
pkl_file.close()

GLOVE_300_DICT = load_glove()

def find_one_pun(word):
    best_match = ''
    best_score = 0.0
    for text, vector in puns.items():
        match = inner_product(vector, GLOVE_300_DICT[word.lower()])
        if match > best_score:
            best_score = match
            best_match = text
    return best_match, word

def find_top_n_puns(word, n=5):
    score_list = []
    pun_list = []
    pun_return_list = []
    for text, vector in puns.items():
        score = inner_product(vector, GLOVE_300_DICT[word.lower()])
        score_list.append(score)
        pun_list.append(text)
    score_np = np.array(score_list)
    idxs = np.argpartition(score_np, -n)[-n:]
    idxs = idxs[np.argsort([score_list[int(idx)] for idx in idxs])]
    for idx in idxs:
        pun_return_list.append(pun_list[int(idx)])
    pun_return_list.reverse()
    return pun_return_list, word

# print(find_top_n_puns('sports'))