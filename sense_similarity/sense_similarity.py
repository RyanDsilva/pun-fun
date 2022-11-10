from nltk.corpus import wordnet as wn 
from nltk.tag import pos_tag

def map_wordnet_pos(ch):
  if ch == 'n':
    return wn.NOUN
  elif ch == 'v':
    return wn.VERB
  elif ch == 'a':
    return wn.ADVERB
  else:
    return None

def find_synset(word, pos=None):
  if pos is not None:
    senses = wn.synsets(word)
  else:
    senses = wn.synsets(word)
  if senses:
    return senses[0]
  else:
    return None

def find_source_pos(source, sentence):
  res_pos = None
  sent_pos = pos_tag(sentence.split())
  for w, pos in sent_pos:
    if w == source:
      res_pos = pos[0].lower()
  return map_wordnet_pos(res_pos)

def find_sense_similarity(source, target):
  return source.path_similarity(target)