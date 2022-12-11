import nltk
from fuzzywuzzy import fuzz
from sound_similarity_v2.sound_similarity_v2 import get_sound_similarity as gss

arpabet = nltk.corpus.cmudict.dict()
threshold_v1 = 90
threshold = 0.001

def find_similar_sounding_words_v1(source):
  sound = None
  cand_list = []
  try:
    sound = arpabet[source][0] # Picking only the first pronounciation
  except:
    return cand_list
  for w, a in arpabet.items():
    if fuzz.ratio(sound, a) > threshold_v1 and w != source:
      cand_list.append(w)
  return cand_list

def find_similar_sounding_words(source):
  cand_list = []
  stress_diff = 1
  for w, a in arpabet.items():
    # print('arpa')
    d, s1, s2 = gss(source, w)
    # print(d, d_cos, w)
    if d < threshold and abs(s1-s2) < stress_diff and source != w:
      # print(d, d_cos, w)
      cand_list.append(w)
  if cand_list is None:
    cand_list.append(source)
  return cand_list

def get_sound_similarity(source, target):
  src_arp = None
  tgt_arp = None
  try:
    src_arp = arpabet[source][0]
    tgt_arp = arpabet[target][0]
  except:
    return None
  return fuzz.ratio(src_arp, tgt_arp)

# print(find_similar_sounding_words('bare'))