from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL = 'models'

tokenizer = AutoTokenizer.from_pretrained(MODEL, add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(MODEL, num_labels=2)

def identify_source(sentence):
  p = pipeline('token-classification', model=model, tokenizer=tokenizer)
  predictions = p.predict(sentence)
  res = []
  for pred in predictions:
    if pred['entity'] == 'LABEL_1':
      res.append(pred['word'][1:])
  print(' '.join(res))
  if len(res) >= 1:
    return res[0]
  return None