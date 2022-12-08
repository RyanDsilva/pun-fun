import pprint
from setup import setup
import gradio as gr
from target_identification.target_identification import run_inference_pipeline
from sense_similarity.sense_similarity import find_sense_similarity, find_source_pos, find_synset
from sound_similarity.sound_similarity import get_sound_similarity
from source_identification.inference import identify_source

pp = pprint.PrettyPrinter(indent=2)

def run(sentence):
  source_word = identify_source(sentence)
  res = run_inference_pipeline(sentence, source_word)
  for r in res:
    src = r['source']
    pred = r['predicted']
    sent = r['sentence']
    pos = find_source_pos(src, sent)
    src_sense = find_synset(src)
    pred_sense = find_synset(pred)
    sense_sim = None
    if src_sense is not None and pred_sense is not None:
      print("Definitions")
      print(f"{src}: {src_sense.definition()}")
      print(f"{pred}: {pred_sense.definition()}")
      print("=" * 50)
      sense_sim = find_sense_similarity(src_sense, pred_sense)
      print(f"Sense similarity between {src} and {pred} is {sense_sim}")
    sound_sim = get_sound_similarity(src, pred)
    if sound_sim is not None:
      print(f"Sound similarity between {src} and {pred} is {sound_sim}")
    print("=" * 50)
    if sense_sim is not None and sound_sim is not None:
      return source_word, src_sense.definition(), pred, pred_sense.definition(), sense_sim, sound_sim
    elif sense_sim is None and sound_sim is not None:
      return source_word, 'No definition available', pred, 'No definition available', 'NA', sound_sim
    elif sense_sim is not None and sound_sim is None:
      return source_word, src_sense.definition(), pred, pred_sense.definition(), sense_sim, 'NA'
    else:
      return source_word, 'No definition available' , pred, 'No definition available', 'NA', 'NA'

if __name__ == "__main__":
  setup()
  app = gr.Interface(
    title='CNIT 519 - Project 2',
    fn=run,
    allow_flagging='never',
    inputs=[
      gr.Textbox(label='Input Sentence')
    ],
    outputs=[
      gr.Textbox(label='Source'),
      gr.Textbox(label='Source Definition'),
      gr.Textbox(label='Target'),
      gr.Textbox(label='Target Definition'),
      gr.Textbox(label='Sense Similarity'),
      gr.Textbox(label='Sound Similarity'),
    ],
  )
  app.launch(share=True)

