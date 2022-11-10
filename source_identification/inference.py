import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL = 'models'

tokenizer = AutoTokenizer.from_pretrained(MODEL, add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(MODEL, num_labels=2)

# def identify_source(sentence):
#   tokens = sentence.split(" ")
#   tokenized_sent = tokenizer(sentence, truncation=True)
#   predictions, labels, _ = model.predict([tokenized_sent])
#   predictions = np.argmax(predictions, axis=2)[0]
#   locs = [i - 1 for i, e in enumerate(predictions) if e == 1]
#   return tokens[locs][0]

def identify_source(sentence):
  # tokens = sentence.split(" ")
  p = pipeline('token-classification', model=model, tokenizer=tokenizer)
  predictions = p.predict(sentence)
  res = None
  for pred in predictions:
    if pred['entity'] == 'LABEL_1':
      res = pred['word'][1:]
  return res