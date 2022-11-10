import nltk
from fuzzywuzzy import fuzz

arpabet = nltk.corpus.cmudict.dict()
threshold = 90

def find_similar_sounding_words(source):
  sound = None
  cand_list = []
  try:
    sound = arpabet[source][0] # Picking only the first pronounciation
  except:
    return cand_list
  for w, a in arpabet.items():
    if fuzz.ratio(sound, a) > threshold and w != source:
      cand_list.append(w)
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