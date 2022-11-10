import pprint
from setup import setup
from target_identification.target_identification import run_inference_pipeline
from sense_similarity.sense_similarity import find_sense_similarity, find_source_pos, find_synset
from sound_similarity.sound_similarity import get_sound_similarity
from source_identification.inference import identify_source

pp = pprint.PrettyPrinter(indent=2)

if __name__ == "__main__":
  setup()
  sentence = "The boat store had a huge sail ."
  source_word = identify_source(sentence)
  res = run_inference_pipeline(sentence, source_word)
  for r in res:
    src = r['source']
    pred = r['predicted']
    sent = r['sentence']
    pos = find_source_pos(src, sent)
    src_sense = find_synset(src)
    pred_sense = find_synset(pred)
    if src_sense is not None and pred_sense is not None:
      print(f"Sense similarity between {src} and {pred} is {find_sense_similarity(src_sense, pred_sense)}")
    ss = get_sound_similarity(src, pred)
    if ss is not None:
      print(f"Sound similarity between {src} and {pred} is {get_sound_similarity(src, pred)}")
    print("=============================================")

